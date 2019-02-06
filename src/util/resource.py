import datetime
import logging
import pathlib
import socket
import sys

import logzero

from util import const


class Resource(object):

    def __init__(self, args, train=True):
        self.args = args
        self.start_time = datetime.datetime.now()
        self.logger = logzero.setup_default_logger()

        # test=False and train=False
        if not args.test and train:
            self.logger.warn('Test option is {}'.format(args.test))

        # setup experiment directory
        self.output_dir = self._setup_output_dir()
        self.log_dir = self._setup_log_dir()

        if train:
            self.fig_dir = self._setup_fig_dir()
            log_filename = '{}_train.log'.format(self.sdtime)
        else:
            log_filename = '{}_inference.log'.format(self.sdtime)

        log_name = self.log_dir / log_filename

        logzero.logfile(str(log_name), loglevel=logging.INFO)

        self.log_name = log_name
        self.logger.info('Log filename: {}'.format(str(log_name)))
        self.logger.info('Server name: {}'.format(socket.gethostname()))
        self.dump_common_info()

    @property
    def stime(self):
        return self.start_time.strftime('%Y-%m-%d-%H-%M-%S')

    @property
    def sdtime(self):
        return self.start_time.strftime('%H-%M-%S')

    @property
    def sytime(self):
        return self.start_time.strftime('%Y-%m-%d')

    @property
    def duration(self):
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        return duration

    def logdebug(self, msg):
        self.logger.debug(msg)

    def loginfo(self, msg):
        self.logger.info(msg)

    def _setup_output_dir(self):
        output_dir = pathlib.Path(self.args.out) / self.sytime

        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            self.logger.debug('Output dir is created at [{}]'.format(str(output_dir)))
        else:
            self.logger.debug('Output dir is already exists.'.format(str(output_dir)))

        return output_dir

    def _setup_log_dir(self):
        log_dir = self.output_dir / 'log'

        if not log_dir.exists():
            log_dir.mkdir()
            self.logger.debug('Log dir is created at [{}]'.format(str(log_dir)))
        else:
            self.logger.debug('Log dir is already exists.'.format(str(log_dir)))

        return log_dir

    def _setup_fig_dir(self):
        fig_dir = self.output_dir / 'fig'

        self.fig_acc_dir = fig_dir / 'accuracy'
        self.fig_loss_dir = fig_dir / 'loss'
        self.fig_heatmap_dir = fig_dir / 'heatmap'
        self.fig_coef_dir = fig_dir / 'coef'

        if not fig_dir.exists():
            self.logger.debug('Fig dir is created at [{}]'.format(str(fig_dir)))
            fig_dir.mkdir()
            # also make directories
            self.fig_acc_dir.mkdir()
            self.fig_loss_dir.mkdir()
            self.fig_heatmap_dir.mkdir()
            self.fig_coef_dir.mkdir()
        else:
            self.logger.debug('Fig dir is already exists.'.format(str(fig_dir)))

        return fig_dir

    def dump_common_info(self):
        logger = logzero.setup_logger(
            name='preprocess',
            logfile=str(self.log_dir / f'{self.sdtime}_preprocess.log'))

        logger.info('=== Common informations ===')
        logger.info('Model: {}'.format(self.args.arch))
        logger.info('Word embedding: {}'.format(self.args.word_embedding))
        logger.info('Training type: {}'.format(self.args.training_type))

        logger.info('# of epoch: {}'.format(self.args.epoch))
        logger.info('# of batchsize: {}'.format(self.args.batchsize))
        logger.info('# of encoder layers: {}'.format(self.args.layer))
        logger.info('# of genre embedding dim: {}'.format(self.args.genre_unit))
        logger.info('Dropout ratio: {}'.format(self.args.dropout))
        logger.info('Weight decay: {}'.format(self.args.weight_decay))
        logger.info('GPU ID: {}'.format(self.args.gpu))
        logger.info('Cross validation: {}, # of folds: {}'.format(self.args.cv, self.args.fold))
        logger.info('Seed: {}'.format(self.args.seed))
        logger.info('GroupKFold: {}'.format(self.args.group))
        logger.info('Apply impression normalization: {}'.format(self.args.imp_norm))
        logger.info('Target objective: {}'.format(self.args.objective))

    def dump_duration(self):
        logger = logzero.setup_logger(
            name='test',
            logfile=str(self.log_dir / f'{self.sdtime}_test.log')
        )
        end_time = datetime.datetime.now()
        logger.info('Exit time: {}'.format(end_time.strftime('%Y/%m/%d - %H:%M:%S')))
        logger.info('Duration: {}'.format(self.duration))
        logger.info('Remember: log is saved to {}'.format(str(self.log_name)))

    def dump_command_info(self):
        logger = logzero.setup_logger(
            name='test',
            logfile=str(self.log_dir / f'{self.sdtime}_test.log')
        )
        logger.info('Command name: {}'.format(' '.join(sys.argv)))
