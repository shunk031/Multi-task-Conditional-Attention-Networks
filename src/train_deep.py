import matplotlib  # NOQA # isort:skip
matplotlib.use("Agg")  # NOQA # isort:skip

from collections import OrderedDict

import chainer
import logzero
import matplotlib.pyplot as plt
import numpy as np
from chainer import training
from chainer.training import extensions
from gensim.models import KeyedVectors

from models import ARCHS, MODEL_WRAPPERS
from util import const
from util.args import parse_train_args as parse_args
from util.cross_validation import kfold_iter
from util.evaluate import EVALUATE_PHASES
from util.extensions import (
    setup_optim_trigger,
    setup_plot_report_loss_entries,
    setup_print_report_entries,
    setup_record_trigger
)
from util.load import load_data
from util.notify import notify_exception, notify_result
from util.preprocessed_dataset import (
    PreprocessedDataset,
    convert_seq,
    prepare_vectorizer
)
from util.resource import Resource
from util.seed import reset_seed

plt.style.use('ggplot')
chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
chainer.global_config.autotune = True


def main(args):

    reset_seed(args.seed)

    res = Resource(args, train=True)

    pretrained_word2vec = KeyedVectors.load_word2vec_format(
        str(const.PRETRAINED_WORD2VEC_FPATH), binary=True)

    vectorizer = prepare_vectorizer(pretrained_word2vec,
                                    args.training_type,
                                    norm_imp=const.NORMALIZED_IMPRESSION,
                                    is_impnorm=args.imp_norm,
                                    is_logarithm=True)

    df = load_data(const.TRAIN_DATA_FPATH)

    kf = kfold_iter(X=df, y=df[args.objective],
                    n_splits=args.fold,
                    random_state=args.seed,
                    is_campaign_group=args.group,
                    training_type=args.training_type,
                    campaign_ids=df['campaign_id'].values)

    scores = OrderedDict([(metric, []) for metric
                          in const.EVALUATION_METRIC[args.training_type]])

    res.loginfo('Start training')
    for i, (train_idx, val_idx) in enumerate(kf):
        res.loginfo('Fold: {}'.format(i + 1))

        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        df_train = vectorizer.fit_transform(df_train)
        df_val = vectorizer.transform(df_val)

        n_genre = len(vectorizer.named_steps.genre.le.classes_)
        n_gender = len(vectorizer.named_steps.gender.lb.classes_)

        res.logdebug('# of genres: {}'.format(n_genre))
        res.logdebug('# of gender: {}'.format(n_gender))

        train_pairs = PreprocessedDataset(df_train, args.training_type,
                                          output_cols=args.objective)
        val_pairs = PreprocessedDataset(df_val, args.training_type,
                                        output_cols=args.objective)

        train_iter = chainer.iterators.SerialIterator(
            train_pairs, args.batchsize)
        val_iter = chainer.iterators.SerialIterator(
            val_pairs, args.batchsize, repeat=False, shuffle=False)

        is_update_w2v = args.word_embedding == const.WORD2VEC_UPDATE
        encoder = ARCHS[args.arch](n_layers=args.layer,
                                   n_genre=n_genre,
                                   n_vocab=len(pretrained_word2vec.index2word),
                                   pretrained_w2v=pretrained_word2vec,
                                   is_update_w2v=is_update_w2v,
                                   dropout=args.dropout)
        model = MODEL_WRAPPERS[args.training_type](
            encoder=encoder, dropout=args.dropout)

        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

        # Set up a trainer
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=args.gpu,
            converter=convert_seq)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=str(res.output_dir))

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(
            val_iter, model, device=args.gpu,
            converter=convert_seq))

        # Take a best snapshot
        record_trigger = setup_record_trigger(args.training_type)
        optim_trigger = setup_optim_trigger(args.training_type)

        model_fname = '{}_{}_{}-fold_{}_best_model.npz'.format(
            res.sdtime, args.training_type, i + 1, args.arch)
        trainer.extend(extensions.snapshot_object(
            model, model_fname),
            trigger=record_trigger)

        trainer.extend(extensions.ExponentialShift('alpha', 0.9), trigger=optim_trigger)

        # Write a log of evaluation statistics for each epoch
        trainer_log_name = '{}_{}_{}-fold_{}_reporter.json'.format(
            res.sdtime, args.training_type, i + 1, args.arch)
        trainer.extend(extensions.LogReport(log_name=trainer_log_name))
        trainer.extend(extensions.observe_lr())

        fig_loss_fpath = res.fig_loss_dir / '{}_{}_{}-fold_loss.png'.format(
            res.sdtime, args.training_type, i + 1)
        fig_loss_path = fig_loss_fpath.relative_to(res.output_dir)
        plot_loss_entries = setup_plot_report_loss_entries(args.training_type)
        trainer.extend(extensions.PlotReport(plot_loss_entries, 'epoch',
                                             file_name=str(fig_loss_path), grid=False))

        entries = setup_print_report_entries(args.training_type)
        trainer.extend(extensions.PrintReport(entries))

        trainer.extend(extensions.ProgressBar(update_interval=10))

        # Run the training
        trainer.run()

        if args.test:
            res.logger.debug('Start test phase')
            snapshot_best_model = res.output_dir / model_fname
            chainer.serializers.load_npz(str(snapshot_best_model), model)
            res.logdebug('Load: {}'.format(str(snapshot_best_model)))

            calculated_scores = EVALUATE_PHASES[args.training_type](
                res=res, model=model, pairs=val_pairs, fold=i + 1,
                converter=convert_seq, vectorizer=vectorizer).test()

            for metric in scores.keys():
                scores[metric].append(calculated_scores[metric])

        if not args.cv:
            break

    res.dump_command_info()
    logger = logzero.setup_logger(
        name='test',
        logfile=str(res.log_dir / f'{res.sdtime}_test.log'),
    )
    for metric in scores.keys():
        logger.info(f'Average {metric}: {np.mean(scores[metric]):.6f}, var: {np.var(scores[metric]):.6f}')

    res.dump_duration()

    notify_result(res)


if __name__ == '__main__':

    args = parse_args()
    try:
        main(args)
    except Exception as err:
        if not args.debug:
            notify_exception(err)
        raise err
