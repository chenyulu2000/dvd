import torch
import torch.nn as nn

from models.encoders.multimodal_fusion.bimodal_fusion import BimodalFusion


class MultimodalFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.img_ques_fusion = BimodalFusion(
            opt=opt, answer_size=opt.lstm_hidden_size
        )

        self.hist_ques_fusion = BimodalFusion(
            opt=opt, answer_size=opt.lstm_hidden_size
        )

        self.halve_dimension = nn.Linear(
            in_features=opt.lstm_hidden_size * 2,
            out_features=opt.lstm_hidden_size
        )

        self.dropout = nn.Dropout(p=opt.dropout)

        nn.init.kaiming_uniform_(self.halve_dimension.weight)
        nn.init.constant_(self.halve_dimension.bias, 0)

    def forward(self, batch_size, img_features, ques_features, hist_features,
                img_features_mask, ques_features_mask, hist_features_mask):
        img_ques_features = self.img_ques_fusion(
            img_features, ques_features,
            img_features_mask, ques_features_mask
        )
        hist_ques_features = self.hist_ques_fusion(
            hist_features, ques_features,
            hist_features_mask, ques_features_mask
        )
        multimodal_features = torch.cat([img_ques_features, hist_ques_features], dim=1)
        multimodal_features = self.halve_dimension(
            self.dropout(multimodal_features)
        )
        return multimodal_features.view(batch_size, 10, -1)


if __name__ == '__main__':
    from anatool import AnaArgParser

    opt = AnaArgParser().cfg
    mf = MultimodalFusion(opt=opt)
    mf(
        32,
        torch.rand(320, 36, 512, dtype=torch.float),
        torch.rand(320, 20, 512, dtype=torch.float),
        torch.rand(320, 40, 512, dtype=torch.float),
        torch.ones(320, 1, 1, 36, dtype=torch.int),
        torch.ones(320, 1, 1, 20, dtype=torch.int),
        torch.ones(320, 1, 1, 40, dtype=torch.int),
    )
