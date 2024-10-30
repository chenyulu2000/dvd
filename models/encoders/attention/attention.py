import math
import torch
import torch.nn as nn

from ..attention.utils import MLP
from ..attention.utils import LayerNorm


# multi-head attention
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

    def forward(self, v, k, q, mask):
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

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt.fusion_hidden_size
        )

        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        # score.size() (bs*rounds, heads, proposal/len, proposal/len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = nn.functional.softmax(scores, dim=-1)
        # att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# feed forward net
class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=opt.fusion_hidden_size,
            mid_size=opt.fusion_ff_size,
            out_size=opt.fusion_hidden_size,
            dropout_r=opt.fusion_dropout,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class SF(nn.Module):
    def __init__(self, opt):
        super(SF, self).__init__()

        self.mh_att = MHAtt(opt=opt)
        self.ffn = FFN(opt=opt)

        self.dropout1 = nn.Dropout(p=opt.fusion_dropout)
        self.norm1 = LayerNorm(size=opt.fusion_hidden_size)

        self.dropout2 = nn.Dropout(p=opt.fusion_dropout)
        self.norm2 = LayerNorm(size=opt.fusion_hidden_size)

    def forward(self, y, y_mask):
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


class IF(nn.Module):
    def __init__(self, opt):
        super(IF, self).__init__()
        self.mh_att1 = MHAtt(opt=opt)
        self.mh_att2 = MHAtt(opt=opt)
        self.ffn = FFN(opt=opt)

        self.dropout1 = nn.Dropout(p=opt.fusion_dropout)
        self.norm1 = LayerNorm(size=opt.fusion_hidden_size)

        self.dropout2 = nn.Dropout(p=opt.fusion_dropout)
        self.norm2 = LayerNorm(size=opt.fusion_hidden_size)

        self.dropout3 = nn.Dropout(p=opt.fusion_dropout)
        self.norm3 = LayerNorm(size=opt.fusion_hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(
            x + self.dropout1(
                self.mh_att1(v=x, k=x, q=x, mask=x_mask)
            )
        )
        x = self.norm2(
            x + self.dropout2(
                self.mh_att2(v=y, k=y, q=x, mask=y_mask)
            )
        )
        x = self.norm3(
            x + self.dropout3(
                self.ffn(x)
            )
        )
        return x
