import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter


class Regressor(chainer.Chain):

    def __init__(self, encoder, dropout=0.):
        super(Regressor, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.output = L.Linear(encoder.out_units, 1)

        self.dropout = dropout

    def forward(self, ys, **kwargs):

        concat_outputs = F.concat(self.predict(**kwargs), axis=0)
        concat_truths = F.concat(ys, axis=0)

        loss = F.mean_squared_error(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)

        return loss

    def predict(self, **kwargs):

        concat_encodings = F.dropout(self.encoder(**kwargs), ratio=self.dropout)
        concat_outputs = F.sigmoid(self.output(concat_encodings))

        return concat_outputs


class MultiTaskRegressor(chainer.Chain):

    def __init__(self, encoder, dropout=0.):
        super(MultiTaskRegressor, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.output = L.Linear(encoder.out_units, 2)

        self.dropout = dropout

    def forward(self, ys, **kwargs):

        pred_click, pred_cv = self.predict(**kwargs)
        ys = F.stack(ys)
        true_click, true_cv = ys[:, 0], ys[:, 1]

        loss_click = F.mean_squared_error(pred_click, true_click)
        loss_cv = F.mean_squared_error(pred_cv, true_cv)
        loss = loss_click + loss_cv

        reporter.report({'loss': loss.data}, self)
        reporter.report({'loss_click': loss_click.data}, self)
        reporter.report({'loss_cv': loss_cv.data}, self)

        return loss

    def predict(self, **kwargs):

        concat_encodings = F.dropout(self.encoder(**kwargs), ratio=self.dropout)
        output = F.sigmoid(self.output(concat_encodings))
        output_click, output_cv = output[:, 0], output[:, 1]

        return output_click, output_cv
