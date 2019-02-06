import random

import chainer
import numpy as np


def reset_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
