import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler

from util.preprocess import wakati


class LimitDataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, num_imp):
        self.num_imp = num_imp

    def fit(self, X, *args):
        return self

    def transform(self, df):
        df = df[df['impression'] > self.num_imp]
        return df


class WakatiTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column, wakati_func=wakati):
        self.column = column
        self.wakati_func = wakati

    def fit(self, X, *args):
        return self

    def transform(self, X):
        X[self.column] = X[self.column].apply(self.wakati_func)
        return X


class Word2VecTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns, pretrain_w2v):
        self.columns = columns
        self.w2v = pretrain_w2v
        self.w2i = {w: i for i, w in enumerate(pretrain_w2v.index2word)}

    def fit(self, X, *args):
        return self

    def word2id(self, words):
        return np.asarray([
            self.w2i[w] for w in words
            if w in self.w2v.vocab], dtype=np.int32)

    def transform(self, X):
        for col in self.columns:
            X[col] = X[col].apply(self.word2id)
        return X


class GenreTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.le = None

    def fit(self, X, *args):
        self.le = LabelEncoder().fit(X['genre'])
        return self

    def transform(self, X):
        X['genre'] = self.le.transform(X['genre'])
        return X


class GenderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.lb = None

    def fit(self, X, *args):
        self.lb = LabelBinarizer().fit(X['gender_target'])
        return self

    def transform(self, X):
        X['gender_target'] = self.lb.transform(X['gender_target']).tolist()
        X['gender_target'] = X['gender_target'].apply(
            lambda x: np.asarray(x, dtype=np.float32))
        return X


class MinMaxScaleTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column
        self.mm = None

    def fit(self, X, *args):
        self.mm = MinMaxScaler().fit(X[[self.column]])
        return self

    def transform(self, X):
        X[self.column] = self.mm.transform(X[[self.column]])
        return X


class ToLogarithmTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, *args):
        return self

    def transform(self, X):
        X[self.column] = np.log1p(X[self.column])
        return X

    def inverse_transform(self, X):
        X[self.colum] = np.expm1(X[self.column])
        return X


class TypeConvertTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns, dtype):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, *args):
        return self

    def transform(self, X):
        for col in self.columns:
            X[col] = X[col].astype(self.dtype)
        return X
