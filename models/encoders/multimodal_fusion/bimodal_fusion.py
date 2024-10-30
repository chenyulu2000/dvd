import torch.nn as nn
import torch

from models.encoders.attention.attention import IF
from models.encoders.attention.utils import MLP, LayerNorm


class AttFlat(nn.Module):
    def __init__(self, opt):
        super(AttFlat, self).__init__()
        self.opt = opt

        self.mlp = MLP(
            in_size=opt.fusion_hidden_size,
            mid_size=opt.fusion_flat_mlp_size,
            out_size=opt.fusion_flat_glimpses,
            dropout_r=opt.fusion_dropout,
            use_relu=True
        )
        # FLAT_GLIMPSES == 1
        self.linear_merge = nn.Linear(
            in_features=opt.fusion_hidden_size * opt.fusion_flat_glimpses,
            out_features=opt.fusion_flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = nn.functional.softmax(att, dim=1)

        att_list = []
        for i in range(self.opt.fusion_flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
            #  MLP attention wise sum for each example --> note dim is 1 here

            # z = torch.sum(att[:, :, i: i + 1] * x, dim=1)
            # print(z.size())  # (bs*round, emb_size)

            x_atted = torch.cat(att_list, dim=1)
            # print(x_atted.size())  # (bs*round, emb_size)
            x_atted = self.linear_merge(x_atted)

            return x_atted


class BimodalFusion(nn.Module):
    def __init__(self, opt, answer_size):
        super().__init__()
        self.opt = opt

        self.inter_fusion_list = nn.ModuleList(
            [IF(opt=opt) for _ in range(opt.fusion_layer)]
        )

        self.x_attflat = AttFlat(opt=opt)
        self.y_attflat = AttFlat(opt=opt)

        self.projection_norm = LayerNorm(size=opt.fusion_flat_out_size)
        self.projection = nn.Linear(
            in_features=opt.fusion_flat_out_size,
            out_features=answer_size
        )

    def forward(self, x, y, x_mask, y_mask):
        for inter_fusion in self.inter_fusion_list:
            x = inter_fusion(x, y, x_mask, y_mask)
        x = self.x_attflat(x, x_mask)
        y = self.y_attflat(y, y_mask)

        return self.projection(
            self.projection_norm(
                x + y
            )
        )
