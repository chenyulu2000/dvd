import torch
import torch.nn as nn

from models.encoders.attention.attention import SF
from models.encoders.attention.utils import make_mask


class ImageEncoder(nn.Module):
    def __init__(self, opt, disentangle=False):
        super().__init__()
        self.opt = opt
        self.disentangle = disentangle
        self.feature_projection = nn.Linear(
            in_features=opt.img_feature_size,
            out_features=opt.lstm_hidden_size
        )

        self.self_fusion_list = nn.ModuleList(
            [SF(opt=opt) for _ in range(opt.fusion_layer)]
        )

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
        for self_fusion in self.self_fusion_list:
            img_features = self_fusion(img_features, img_features_mask)
        if self.disentangle:
            img_features = self.halve_dimension(img_features)
        return img_features, img_features_mask


if __name__ == '__main__':
    from anatool import AnaArgParser

    opt = AnaArgParser().cfg
    x = torch.rand(32, 36, 2048)
    ie = ImageEncoder(opt)
    f, m = ie(x)
    print(f.shape, m.shape)
