import torch
import torch.nn as nn

from models.dynamic_rnn import DynamicRNN
from models.encoders.attention.attention import SF
from models.encoders.attention.utils import make_mask


class HistoryEncoder(nn.Module):
    def __init__(self, opt, shared_word_embed, disentangle=False):
        super().__init__()
        self.opt = opt

        self.word_embed = shared_word_embed
        self.disentangle = disentangle

        self.rnn = DynamicRNN(rnn_model=nn.LSTM(
            input_size=opt.word_embedding_size,
            hidden_size=opt.lstm_hidden_size,
            num_layers=opt.lstm_num_layers,
            batch_first=True,
            dropout=opt.dropout
        ))

        self.self_fusion_list = nn.ModuleList(
            [SF(opt=opt) for _ in range(opt.fusion_layer)]
        )

        if disentangle:
            self.halve_dimension = nn.Linear(
                in_features=opt.lstm_hidden_size,
                out_features=opt.lstm_hidden_size // 2
            )
            nn.init.kaiming_uniform_(self.halve_dimension.weight)
            nn.init.constant_(self.halve_dimension.bias, 0)

    def forward(self, cap, cap_length, hist, hist_length):
        batch_size, num_rounds, max_sequence_length = hist.shape
        hist_length = hist_length.unsqueeze(-1).view(
            batch_size, num_rounds, -1
        )
        cap_length = cap_length.unsqueeze(-1).unsqueeze(-1).view(
            batch_size, 1, 1
        )
        hist, hist_length = hist[:, :-1, :], hist_length[:, :-1, :]
        cap = cap.unsqueeze(1).view(
            batch_size, 1, -1
        )
        hist = torch.cat([cap, hist], dim=1)
        hist_length = torch.cat([cap_length, hist_length], dim=1)
        hist = hist.view(batch_size * num_rounds, -1)
        hist_embed = self.word_embed(hist.long())
        hist_features, _ = self.rnn(hist_embed, hist_length)

        hist_features_mask = make_mask(hist_features)
        for self_fusion in self.self_fusion_list:
            hist_features = self_fusion(hist_features, hist_features_mask)
        if self.disentangle:
            hist_features = self.halve_dimension(hist_features)
        return hist_features, hist_features_mask, hist_length


if __name__ == '__main__':
    from anatool import AnaArgParser, AnaLogger
    from data.vocabulary import Vocabulary

    opt = AnaArgParser().cfg
    logger = AnaLogger()
    vocab = Vocabulary(
        word_counts_path=opt.word_counts_json,
        min_count=opt.vocab_min_count,
        logger=logger
    )
    shared_word_embed = nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=opt.word_embedding_size,
        padding_idx=vocab.PAD_INDEX
    )
    hist = torch.rand(32, 10, 40, dtype=torch.float)
    hist_len = torch.ones(32, 10, 1, dtype=torch.int)
    cap = torch.rand(32, 40, dtype=torch.float)
    cap_len = torch.ones(32, 1, 1, dtype=torch.int)
    he = HistoryEncoder(opt=opt, shared_word_embed=shared_word_embed, disentangle=True)
    he(cap, cap_len, hist, hist_len)
