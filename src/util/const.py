import pathlib

ROOT_DIR = pathlib.Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / 'data'

TRAIN_DATA_FNAME = 'train.csv'
TRAIN_DATA_FPATH = DATA_DIR / TRAIN_DATA_FNAME

PRETRAINED_WORD2VEC_FNAME = 'entity_vector.model.bin'
PRETRAINED_WORD2VEC_FPATH = DATA_DIR / PRETRAINED_WORD2VEC_FNAME

NEOLOGD_DIR = '/var/lib/mecab/dic/mecab-ipadic-neologd'

DATASET_FROM_DATE = '2017-08-01'
DATASET_TO_DATE = '2018-08-01'

PREPROCESS_GENDER_TARGET_NUM = 3

NORMALIZED_IMPRESSION = 30000

TRAINING_TYPES_REGRESSION = 'regression'
TRAINING_TYPES_MULTI_REGRESSION = 'multi_regression'
TRAINING_TYPES = [
    TRAINING_TYPES_REGRESSION,
    TRAINING_TYPES_MULTI_REGRESSION,
]

WORD2VEC_UPDATE = 'word2vec_update'
WORD2VEC_FREEZE = 'word2vec_freeze'
FROM_SCRATCH = 'scratch'
WORD2VEC_TYPES = [
    WORD2VEC_UPDATE,
    WORD2VEC_FREEZE,
    FROM_SCRATCH,
]

EVALUATION_METRIC = {
    TRAINING_TYPES_REGRESSION: [
        'MSE_CV', 'MSE_gt_1', 'MAP_0', 'MAP_10', 'NDCG_CV',
        'MSE_CV_top_50', 'MSE_CV_top_25', 'MSE_CV_top_10', 'MSE_CV_top_5', 'MSE_CV_top_1',
        'NDCG_CV_top_50', 'NDCG_CV_top_25', 'NDCG_CV_top_10', 'NDCG_CV_top_5', 'NDCG_CV_top_1',
    ],
    TRAINING_TYPES_MULTI_REGRESSION: [
        'MSE_click', 'MSE_CV', 'MSE_multi', 'MSE_CVR', 'MAP_0', 'MAP_10', 'MAP_CVR', 'NDCG_CV', 'NDCG_CVR',
        'MSE_CV_top_50', 'MSE_CV_top_25', 'MSE_CV_top_10', 'MSE_CV_top_5', 'MSE_CV_top_1',
        'NDCG_CV_top_50', 'NDCG_CV_top_25', 'NDCG_CV_top_10', 'NDCG_CV_top_5', 'NDCG_CV_top_1',
    ],
}
