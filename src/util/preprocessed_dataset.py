import chainer
import numpy as np
from chainer import cuda
from sklearn.pipeline import Pipeline

from util import const
from util.transforms import (
    GenderTransformer,
    GenreTransformer,
    LimitDataTransformer,
    MinMaxScaleTransformer,
    ToLogarithmTransformer,
    TypeConvertTransformer,
    Word2VecTransformer
)


def prepare_vectorizer(pretrain_w2v,
                       training_type,
                       norm_imp=None,
                       is_impnorm=False,
                       is_logarithm=False,
                       ):
    steps = [
        ('lmit', LimitDataTransformer()),
        ('genre', GenreTransformer()),
        ('gender', GenderTransformer()),
        ('w2v', Word2VecTransformer(
            columns=['title_text', 'content_text'],
            pretrain_w2v=pretrain_w2v)),
    ]

    if is_logarithm:
        steps.extend([
            ('log_click', ToLogarithmTransformer(column='click')),
            ('loc_cv', ToLogarithmTransformer(column='conversion')),
        ])

    if training_type in [const.TRAINING_TYPES_REGRESSION,
                         const.TRAINING_TYPES_MULTI_REGRESSION]:
        if is_impnorm:
            click_col = 'imp_norm_click'
            cv_col = 'imp_norm_conversion'
        else:
            click_col = 'click'
            cv_col = 'conversion'

        scaler = [
            ('mm_click', MinMaxScaleTransformer(column=click_col)),
            ('mm_conversion', MinMaxScaleTransformer(column=cv_col)),
            ('mm_cvr', MinMaxScaleTransformer(column='cvr')),
        ]
        steps.extend(scaler)

    steps.extend([
        ('type_convert', TypeConvertTransformer(
            columns=['product_id', 'genre'],
            dtype=np.int32)),
    ])

    return Pipeline(steps)


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, df,
                 training_type,
                 output_cols):

        self.X = df.drop(output_cols, axis=1)
        self.y = df[output_cols]

        self.training_type = training_type

    def __len__(self):
        return len(self.X)

    def get_example(self, i):

        X = self.X.iloc[i]
        y = self.y.iloc[i]

        X = X.to_dict()
        y = y.values.astype(np.float32)

        return X, y


def convert_seq(batch, device=None, with_label=True):

    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                  for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    if with_label:
        return {
            'product_xs': to_device_batch([[x['product_id']] for x, _ in batch]),
            'genre_xs': to_device_batch([np.asarray([x['genre']]) for x, _ in batch]),
            'gender_xs': to_device_batch([x['gender_target'] for x, _ in batch]),
            'title_xs': to_device_batch([x['title_text'] for x, _ in batch]),
            'content_xs': to_device_batch([x['content_text'] for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch]),
        }

    else:
        return to_device_batch([x for x in batch])
