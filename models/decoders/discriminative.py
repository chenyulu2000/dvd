import torch
import torch.nn as nn

from models.dynamic_rnn import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    def __init__(self, opt, shared_word_embed):
        super().__init__()
        self.opt = opt

        self.word_embed = shared_word_embed

        self.rnn = DynamicRNN(rnn_model=nn.LSTM(
            input_size=opt.word_embedding_size,
            hidden_size=opt.lstm_hidden_size,
            num_layers=opt.lstm_num_layers,
            batch_first=True,
            dropout=opt.dropout
        ))

    def forward(self, fusion_output, options, options_length):
        batch_size, num_rounds, num_options, max_sequence_length = options.shape
        options = options.view(batch_size * num_rounds * num_options, max_sequence_length)
        options_length = options_length.view(batch_size * num_rounds * num_options)

        # pick options with non-zero length (relevant for test split).
        nonzero_options_length_indices = options_length.nonzero().squeeze()
        nonzero_options_length = options_length[nonzero_options_length_indices]
        nonzero_options = options[nonzero_options_length_indices]

        nonzero_options_embed = self.word_embed(nonzero_options)
        _, (nonzero_options_embed, _) = self.rnn(
            nonzero_options_embed, nonzero_options_length
        )

        options_embed = torch.zeros(
            batch_size * num_rounds * num_options,
            nonzero_options_embed.size(-1),
            device=nonzero_options_embed.device
        )
        options_embed[nonzero_options_length_indices] = nonzero_options_embed

        # reapeat encoder output for every option.
        # shape: (batch_size, num_rounds, num_options, max_sequence_length)
        fusion_output = fusion_output.unsqueeze(2).repeat(
            1, 1, num_options, 1
        )

        # shape now same as 'options', can calculate dot producat similarity.
        # shape: (batch_size * num_rounds * num_options, lstm_hidden_state)
        fusion_output = fusion_output.view(
            batch_size * num_rounds * num_options,
            self.opt.lstm_hidden_size
        )

        # shape: (batch_size * num_rounds *num_options)
        scores = torch.sum(options_embed * fusion_output, 1)
        # shape: (batch_size, num_rounds, num_options)
        scores = scores.view(batch_size, num_rounds, num_options)
        return scores
