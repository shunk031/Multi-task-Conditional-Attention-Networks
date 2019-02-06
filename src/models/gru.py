import chainer
import chainer.functions as F
import chainer.links as L

from models.base_embedding_layer import BaseEmbeddingLayer
from util import const


class GRUEncoder(chainer.Chain):

    def __init__(self,
                 n_layers,
                 n_vocab,
                 n_genre,
                 pretrained_w2v,
                 is_update_w2v,
                 dropout,
                 genre_units=5):

        super(GRUEncoder, self).__init__()
        with self.init_scope():
            self.base_embedding_layer = BaseEmbeddingLayer(
                n_vocab=n_vocab,
                n_genre=n_genre, genre_units=genre_units,
                pretrained_w2v=pretrained_w2v,
                is_update_w2v=is_update_w2v,
                dropout=dropout)

            self.title_encoder = L.NStepGRU(
                n_layers,
                self.base_embedding_layer.n_units,
                self.base_embedding_layer.n_units, dropout)
            self.content_encoder = L.NStepGRU(
                n_layers,
                self.base_embedding_layer.n_units,
                self.base_embedding_layer.n_units, dropout)

        self.out_units = self.base_embedding_layer.n_units * 2 \
            + genre_units \
            + const.PREPROCESS_GENDER_TARGET_NUM

        self.n_layers = n_layers
        self.dropout = dropout

    def forward(self,
                genre_xs,
                gender_xs,
                title_xs,
                content_xs,
                **kwargs):

        embeddings = self.base_embedding_layer(
            title_xs=title_xs, content_xs=content_xs,
            genre_xs=genre_xs)
        title_exs, content_exs, genre_exs = embeddings
        gender_exs = F.stack(gender_xs)

        last_title_h, title_ys = self.title_encoder(None, title_exs)
        last_content_h, content_ys = self.content_encoder(None, content_exs)

        concat_outputs = F.concat((
            genre_exs,
            gender_exs,
            last_title_h[-1],
            last_content_h[-1],
        ))

        return concat_outputs


class AttentionGRUEncoder(GRUEncoder):

    def __init__(self,
                 n_layers,
                 n_vocab,
                 n_genre,
                 pretrained_w2v,
                 is_update_w2v,
                 dropout,
                 genre_units=5):

        super(AttentionGRUEncoder, self).__init__(
            n_layers=n_layers,
            n_vocab=n_vocab,
            n_genre=n_genre,
            pretrained_w2v=pretrained_w2v,
            is_update_w2v=is_update_w2v,
            dropout=dropout,
            genre_units=genre_units)

        with self.init_scope():
            self.attn_title = L.Linear(self.base_embedding_layer.n_units, 1)
            self.attn_content = L.Linear(self.base_embedding_layer.n_units, 1)

    def calc_attention(self, xs, ys, attn_linear):

        concat_ys = F.concat(ys, axis=0)
        attn_ys = attn_linear(F.tanh(concat_ys))

        cumsum_ys = self.xp.cumsum(self.xp.array([len(x) for x in xs], dtype=self.xp.int32))

        split_attn_ys = F.split_axis(attn_ys, cumsum_ys[:-1].tolist(), axis=0)
        split_attn_ys_pad = F.pad_sequence(split_attn_ys, padding=-1024)
        attn_softmax = F.softmax(split_attn_ys_pad, axis=1)

        return attn_softmax

    def apply_attention(self, ys, attn_softmax):
        batchsize = len(ys)

        ys_pad = F.pad_sequence(ys, padding=0.0)
        ys_pad_reshape = F.reshape(ys_pad, (-1, ys_pad.shape[-1]))

        attn_softmax_reshape = F.broadcast_to(
            F.reshape(attn_softmax, (-1, attn_softmax.shape[-1])), ys_pad_reshape.shape)

        attn_hidden = ys_pad_reshape * attn_softmax_reshape
        attn_hidden_reshape = F.reshape(attn_hidden, (batchsize, -1, attn_hidden.shape[-1]))

        return F.sum(attn_hidden_reshape, axis=1)

    def forward(self,
                genre_xs,
                gender_xs,
                title_xs,
                content_xs,
                **kwargs):

        embedding = self.base_embedding_layer(
            title_xs=title_xs, content_xs=content_xs,
            genre_xs=genre_xs)
        title_exs, content_exs, genre_exs = embedding
        gender_exs = F.stack(gender_xs)

        last_title_h, title_ys = self.title_encoder(None, title_exs)
        last_content_h, content_ys = self.content_encoder(None, content_exs)

        attn_title = self.calc_attention(title_xs, title_ys, self.attn_title)
        attn_title_h = self.apply_attention(title_ys, attn_title)

        attn_content = self.calc_attention(content_xs, content_ys, self.attn_content)
        attn_content_h = self.apply_attention(content_ys, attn_content)

        concat_outputs = F.concat((
            genre_exs,
            gender_exs,
            attn_title_h,
            attn_content_h,
        ))

        return concat_outputs


class ConditionalAttentionGRUEncoder(AttentionGRUEncoder):

    def __init__(self,
                 n_layers,
                 n_vocab,
                 n_genre,
                 pretrained_w2v,
                 is_update_w2v,
                 dropout,
                 genre_units=5):

        super(ConditionalAttentionGRUEncoder, self).__init__(
            n_layers=n_layers,
            n_vocab=n_vocab,
            n_genre=n_genre,
            pretrained_w2v=pretrained_w2v,
            is_update_w2v=is_update_w2v,
            dropout=dropout,
            genre_units=genre_units)

        with self.init_scope():
            self.proj_cond = L.Linear(None, 1, nobias=True)

    def calc_attention(self, xs, ys, genre_exs, gender_exs, attn_linear):

        concat_ys = F.concat(ys, axis=0)  # -> (total len of batched sentence, word embedding dim)
        attn_ys = attn_linear(F.tanh(concat_ys))
        cond_feature = self.proj_cond(F.concat((genre_exs, gender_exs)))  # -> (batchsize, proj_cond dim)

        cumsum_ys = self.xp.cumsum(self.xp.array([len(x) for x in xs], dtype=self.xp.int32))
        split_attn_ys = F.split_axis(attn_ys, cumsum_ys[:-1].tolist(), axis=0)
        split_attn_ys_pad = F.pad_sequence(split_attn_ys, padding=-1024)

        bool_cond = split_attn_ys_pad.array == -1024
        split_attn_ys_pad = split_attn_ys_pad * F.expand_dims(
            F.broadcast_to(cond_feature, (split_attn_ys_pad.shape[:-1])), axis=-1)

        padding_array = self.xp.full(split_attn_ys_pad.shape, -1024, dtype=self.xp.float32)

        split_attn_ys_pad = F.where(bool_cond, padding_array, split_attn_ys_pad)

        attn_softmax = F.softmax(split_attn_ys_pad, axis=1)

        return attn_softmax

    def apply_attention(self, ys, attn_softmax):
        batchsize = len(ys)

        ys_pad = F.pad_sequence(ys, padding=0.0)
        ys_pad_reshape = F.reshape(ys_pad, (-1, ys_pad.shape[-1]))

        attn_softmax_reshape = F.broadcast_to(
            F.reshape(attn_softmax, (-1, attn_softmax.shape[-1])), ys_pad_reshape.shape)

        attn_hidden = ys_pad_reshape * attn_softmax_reshape
        attn_hidden_reshape = F.reshape(attn_hidden, (batchsize, -1, attn_hidden.shape[-1]))

        return F.sum(attn_hidden_reshape, axis=1)

    def forward(self,
                genre_xs,
                gender_xs,
                title_xs,
                content_xs,
                **kwargs):

        embedding = self.base_embedding_layer(
            title_xs=title_xs, content_xs=content_xs,
            genre_xs=genre_xs)
        title_exs, content_exs, genre_exs = embedding
        gender_exs = F.stack(gender_xs)

        last_title_h, title_ys = self.title_encoder(None, title_exs)
        last_content_h, content_ys = self.content_encoder(None, content_exs)

        attn_title = self.calc_attention(title_xs, title_ys, genre_exs,
                                         gender_exs, self.attn_title)
        attn_title_h = self.apply_attention(title_ys, attn_title)

        attn_content = self.calc_attention(content_xs, content_ys, genre_exs,
                                           gender_exs, self.attn_content)
        attn_content_h = self.apply_attention(content_ys, attn_content)

        concat_outputs = F.concat((
            genre_exs,
            gender_exs,
            attn_title_h,
            attn_content_h,
        ))

        return concat_outputs
