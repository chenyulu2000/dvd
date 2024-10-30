import torch
import torch.nn as nn

from models.encoders.attention.attention import FFN
from models.encoders.attention.utils import make_mask, LayerNorm
import math


class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(
            in_features=opt.fusion_hidden_size,
            out_features=opt.fusion_hidden_size
        )
        self.linear_k = nn.Linear(
            in_features=opt.fusion_hidden_size,
            out_features=opt.fusion_hidden_size
        )
        self.linear_q = nn.Linear(
            in_features=opt.fusion_hidden_size,
            out_features=opt.fusion_hidden_size
        )
        self.linear_merge = nn.Linear(
            in_features=opt.fusion_hidden_size,
            out_features=opt.fusion_hidden_size
        )
        self.dropout = nn.Dropout(p=opt.fusion_dropout)

    def forward(self, v, k, q, mask, only_attention=False):
        n_batches = q.size(0)

        # v.size() (bs*rounds, proposal/len, emb_size)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt.fusion_multi_head,
            int(self.opt.fusion_hidden_size / self.opt.fusion_multi_head)
        ).transpose(1, 2)
        # v.size() (bs*rounds, heads, proposal/len, emb_size)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt.fusion_multi_head,
            int(self.opt.fusion_hidden_size / self.opt.fusion_multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt.fusion_multi_head,
            int(self.opt.fusion_hidden_size / self.opt.fusion_multi_head)
        ).transpose(1, 2)

        if only_attention:
            return self.att(v, k, q, mask, True)
        else:
            atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt.fusion_hidden_size
        )

        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask, only_attention=False):
        d_k = query.size(-1)

        # score.size() (bs*rounds, heads, proposal/len, proposal/len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        if only_attention:
            return scores
        else:
            att_map = nn.functional.softmax(scores, dim=-1)
            return torch.matmul(att_map, value)


class SF(nn.Module):
    def __init__(self, opt):
        super(SF, self).__init__()
        self.mh_att = MHAtt(opt=opt)
        self.ffn = FFN(opt=opt)

        self.dropout1 = nn.Dropout(p=opt.fusion_dropout)
        self.norm1 = LayerNorm(size=opt.fusion_hidden_size)

        self.dropout2 = nn.Dropout(p=opt.fusion_dropout)
        self.norm2 = LayerNorm(size=opt.fusion_hidden_size)

    def forward(self, y, y_mask, only_attention=False):
        if not only_attention:
            y = self.norm1(
                y + self.dropout1(
                    self.mh_att(v=y, k=y, q=y, mask=y_mask)
                )
            )
            y = self.norm2(
                y + self.dropout2(
                    self.ffn(y)
                )
            )
            return y
        else:
            return self.mh_att(v=y, k=y, q=y, mask=y_mask, only_attention=only_attention)


class ImageEncoderOnlyAttention(nn.Module):
    def __init__(self, opt, disentangle=False):
        super().__init__()
        self.opt = opt
        self.disentangle = disentangle
        self.feature_projection = nn.Linear(
            in_features=opt.img_feature_size,
            out_features=opt.lstm_hidden_size
        )

        self.self_fusion_list = nn.ModuleList([SF(opt=opt) for _ in range(opt.fusion_layer)])

        nn.init.kaiming_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0)

        if disentangle:
            self.halve_dimension = nn.Linear(
                in_features=opt.lstm_hidden_size,
                out_features=opt.lstm_hidden_size // 2
            )
            nn.init.kaiming_uniform_(self.halve_dimension.weight)
            nn.init.constant_(self.halve_dimension.bias, 0)

    def forward(self, img_features):
        batch_size, num_rounds = len(img_features), 10
        img_features = self.feature_projection(img_features)
        img_features = img_features.view(
            batch_size, 1, -1, self.opt.lstm_hidden_size
        ).repeat(1, num_rounds, 1, 1).view(batch_size * num_rounds, -1, self.opt.lstm_hidden_size)

        img_features_mask = make_mask(feature=img_features)
        for layer, self_fusion in enumerate(self.self_fusion_list):
            img_features = self_fusion(img_features, img_features_mask, (layer + 1) == self.opt.fusion_layer)
        return img_features
