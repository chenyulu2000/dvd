import torch
import torch.nn as nn

from models.decoders.discriminative import DiscriminativeDecoder
from models.decoders.generative import GenerativeDecoder
from models.encoders.disentanglement import HistoryEncoder, QuestionEncoder, ImageEncoder
from models.encoders.multimodal_fusion.multimodal_fusion import MultimodalFusion


class BaseModel(nn.Module):
    def __init__(self, opt, shared_word_embed):
        super().__init__()
        self.history_encoder = HistoryEncoder(
            opt=opt,
            shared_word_embed=shared_word_embed
        )
        self.question_encoder = QuestionEncoder(
            opt=opt,
            shared_word_embed=shared_word_embed
        )
        self.image_encoder = ImageEncoder(opt=opt)

        self.multimodal_fusion = MultimodalFusion(opt=opt)

        if opt.decoder == 'disc':
            self.decoder = DiscriminativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed
            )
        else:
            self.decoder = GenerativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed,
            )

    def forward(self, batch):
        batch_size = batch['img_feat'].size(0)
        img_features, img_features_mask = self.image_encoder(
            img_features=batch['img_feat']
        )
        ques_features, ques_features_mask, ques_length = self.question_encoder(
            ques=batch['ques'],
            ques_length=batch['ques_len']
        )
        hist_features, hist_features_mask, hist_length = self.history_encoder(
            cap=batch['cap'],
            cap_length=batch['cap_len'],
            hist=batch['hist'],
            hist_length=batch['hist_len']
        )
        multimodal_features = self.multimodal_fusion(
            batch_size=batch_size,
            img_features=img_features,
            ques_features=ques_features,
            hist_features=hist_features,
            img_features_mask=img_features_mask,
            ques_features_mask=ques_features_mask,
            hist_features_mask=hist_features_mask
        )
        if isinstance(self.decoder, DiscriminativeDecoder):
            return self.decoder(
                fusion_output=multimodal_features,
                options=batch['opt'],
                options_length=batch['opt_len']
            )
        else:
            if self.training:
                return self.decoder(multimodal_features, batch['ans_in'])
            else:
                return self.decoder(multimodal_features, batch['opt_in'], batch['opt_out'])


class DebiasHistoryModel(nn.Module):
    def __init__(self, opt, shared_word_embed):
        super().__init__()
        self.opt = opt
        self.multimodal_fusion = MultimodalFusion(opt=opt)
        if opt.decoder == 'disc':
            self.decoder = DiscriminativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed
            )
        else:
            self.decoder = GenerativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed
            )

    def forward(self, batch_size, img_features, ques_features, hist_features_fg, hist_features_bg,
                img_features_mask, ques_features_mask, hist_features_mask_fg, hist_features_mask_bg, batch):
        hist_features = torch.cat([hist_features_fg, hist_features_bg], dim=-1)
        hist_features_mask = hist_features_mask_fg | hist_features_mask_bg
        multimodal_features = self.multimodal_fusion(
            batch_size=batch_size,
            img_features=img_features,
            ques_features=ques_features,
            hist_features=hist_features,
            img_features_mask=img_features_mask,
            ques_features_mask=ques_features_mask,
            hist_features_mask=hist_features_mask
        )
        if isinstance(self.decoder, DiscriminativeDecoder):
            return self.decoder(multimodal_features, batch['opt'], batch['opt_len'])
        else:
            if self.training:
                return self.decoder(multimodal_features, batch['ans_in'])
            else:
                return self.decoder(multimodal_features, batch['opt_in'], batch['opt_out'])


class DebiasImageModel(nn.Module):
    def __init__(self, opt, shared_word_embed):
        super().__init__()
        self.multimodal_fusion = MultimodalFusion(opt=opt)
        if opt.decoder == 'disc':
            self.decoder = DiscriminativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed
            )
        else:
            self.decoder = GenerativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed
            )

    def forward(self, batch_size, img_features_fg, img_features_bg, ques_features, hist_features,
                img_features_mask_fg, img_features_mask_bg, ques_features_mask, hist_features_mask, batch):
        img_features = torch.cat([img_features_fg, img_features_bg], dim=-1)
        img_features_mask = img_features_mask_fg | img_features_mask_bg
        multimodal_features = self.multimodal_fusion(
            batch_size=batch_size,
            img_features=img_features,
            ques_features=ques_features,
            hist_features=hist_features,
            img_features_mask=img_features_mask,
            ques_features_mask=ques_features_mask,
            hist_features_mask=hist_features_mask
        )
        if isinstance(self.decoder, DiscriminativeDecoder):
            return self.decoder(multimodal_features, batch['opt'], batch['opt_len'])
        else:
            if self.training:
                return self.decoder(multimodal_features, batch['ans_in'])
            else:
                return self.decoder(multimodal_features, batch['opt_in'], batch['opt_out'])


class DebiasModel(nn.Module):
    def __init__(self, opt, shared_word_embed):
        super().__init__()
        self.multimodal_fusion = MultimodalFusion(opt=opt)
        if opt.decoder == 'disc':
            self.decoder = DiscriminativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed
            )
        else:
            self.decoder = GenerativeDecoder(
                opt=opt,
                shared_word_embed=shared_word_embed
            )

    def forward(self, batch_size, img_features_fg, img_features_bg, ques_features, hist_features_fg, hist_features_bg,
                img_features_mask_fg, img_features_mask_bg, ques_features_mask, hist_features_mask_fg,
                hist_features_mask_bg, batch):
        img_features = torch.cat([img_features_fg, img_features_bg], dim=-1)
        img_features_mask = img_features_mask_fg | img_features_mask_bg
        hist_features = torch.cat([hist_features_fg, hist_features_bg], dim=-1)
        hist_features_mask = hist_features_mask_fg | hist_features_mask_bg
        multimodal_features = self.multimodal_fusion(
            batch_size=batch_size,
            img_features=img_features,
            ques_features=ques_features,
            hist_features=hist_features,
            img_features_mask=img_features_mask,
            ques_features_mask=ques_features_mask,
            hist_features_mask=hist_features_mask
        )
        if isinstance(self.decoder, DiscriminativeDecoder):
            return self.decoder(multimodal_features, batch['opt'], batch['opt_len'])
        else:
            if self.training:
                return self.decoder(multimodal_features, batch['ans_in'])
            else:
                return self.decoder(multimodal_features, batch['opt_in'], batch['opt_out'])
