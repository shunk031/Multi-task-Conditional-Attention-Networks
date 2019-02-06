from models import gru
from models.mlp import MLPEncoder
from models.regressor import MultiTaskRegressor, Regressor

ARCHS = {
    'mlp': MLPEncoder,
    'gru': gru.GRUEncoder,
    'gru_attn': gru.AttentionGRUEncoder,
    'gru_attn_cond': gru.ConditionalAttentionGRUEncoder,
    'gru_attn_word_cond': gru.ConditionalWordAttentionGRUEncoder,
}

MODEL_WRAPPERS = {
    'regression': Regressor,
    'multi_regression': MultiTaskRegressor,
}
