import torch
import torch.nn as nn

from models.dynamic_rnn import DynamicRNN
from models.encoders.attention.attention import SF
from models.encoders.attention.utils import make_mask


class QuestionEncoder(nn.Module):
    def __init__(self, opt, shared_word_embed):
        super().__init__()
        self.opt = opt
        self.word_embed = shared_word_embed

        self.rnn = DynamicRNN(rnn_model=nn.LSTM(
            input_size=opt.word_embedding_size,
            hidden_size=opt.lstm_hidden_size,
            num_layers=opt.lstm_num_layers,
            batch_first=True
        ))

        self.self_fusion_list = nn.ModuleList(
            [SF(opt=opt) for _ in range(opt.fusion_layer)]
        )

    def forward(self, ques, ques_length):
        batch_size, num_rounds, max_sequence_length = ques.shape
        ques = ques.view(
            batch_size * num_rounds,
            max_sequence_length
        )
        ques_features_mask = make_mask(feature=ques.unsqueeze(2))
        ques = ques.squeeze(1)
        ques_embed = self.word_embed(ques.long())
        ques_features, _ = self.rnn(ques_embed, ques_length)

        for self_fusion in self.self_fusion_list:
            ques_features = self_fusion(ques_features, ques_features_mask)
        return ques_features, ques_features_mask, ques_length


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
    ques = torch.rand(32, 10, 20, dtype=torch.float)
    ques_len = torch.ones(32, 10, 1, dtype=torch.int)
    qe = QuestionEncoder(opt=opt, shared_word_embed=shared_word_embed)
    qe(ques, ques_len)
