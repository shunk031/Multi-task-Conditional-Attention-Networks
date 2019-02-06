import chainer.functions as F
import numpy as np


def sequence_embed(embed, xs, dropout=0.):

    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])

    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):

    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e
