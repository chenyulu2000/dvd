import torch
import torch.nn as nn


class GenerativeDecoder(nn.Module):
    def __init__(self, opt, shared_word_embed):
        super().__init__()

        self.opt = opt

        self.word_embed = shared_word_embed

        self.rnn = nn.LSTM(
            input_size=opt.word_embedding_size,
            hidden_size=opt.lstm_hidden_size,
            num_layers=opt.lstm_num_layers,
            batch_first=True,
            dropout=opt.dropout
        )

        self.lstm_to_words = nn.Linear(
            in_features=opt.lstm_hidden_size,
            out_features=self.word_embed.num_embeddings
        )

        self.dropout = nn.Dropout(p=opt.dropout)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_output, ans_in, opt_out=None):
        if self.training:
            batch_size, num_rounds, max_sequence_length = ans_in.size()

            ans_in = ans_in.view(batch_size * num_rounds, max_sequence_length)

            ans_in_embed = self.word_embed(ans_in)
            # shape: (lstm_num_layers, batch_size * num_rounds, lstm_hidden_size)
            init_hidden = encoder_output.view(1, batch_size * num_rounds, -1)
            init_hidden = init_hidden.repeat(self.opt.lstm_num_layers, 1, 1)
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds, max_sequence_length, lstm_hidden_size)
            ans_out, _ = self.rnn(ans_in_embed, (init_hidden, init_cell))
            ans_out = self.dropout(ans_out)

            # shape: (batch_size * num_rounds, max_sequence_length, vocabulary_size)
            ans_word_scores = self.lstm_to_words(ans_out)
            return ans_word_scores
        else:
            batch_size, num_rounds, num_options, max_sequence_length = ans_in.size()

            ans_in = ans_in.view(batch_size * num_rounds * num_options, max_sequence_length)

            # shape: (batch_size * num_rounds * num_options, max_sequence_length
            #         word_embedding_size)
            ans_in_embed = self.word_embed(ans_in)

            # reshape encoder output to be set as initial hidden state of LSTM.
            # shape: (lstm_num_layers, batch_size * num_rounds * num_options,
            #         lstm_hidden_size)
            init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)
            init_hidden = init_hidden.repeat(1, 1, num_options, 1)
            init_hidden = init_hidden.view(1, batch_size * num_rounds * num_options, -1)
            init_hidden = init_hidden.repeat(self.opt.lstm_num_layers, 1, 1)
            init_cell = torch.zeros_like(init_hidden)

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, lstm_hidden_size)
            ans_out, _ = self.rnn(ans_in_embed, (init_hidden, init_cell))

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length, vocabulary_size)
            ans_word_scores = self.logsoftmax(self.lstm_to_words(ans_out))

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            target_ans_out = opt_out.view(batch_size * num_rounds * num_options, -1)

            # shape: (batch_size * num_rounds * num_options,
            #         max_sequence_length)
            ans_word_scores = torch.gather(
                ans_word_scores, -1, target_ans_out.unsqueeze(-1)
            ).squeeze()
            ans_word_scores = (ans_word_scores * (target_ans_out > 0).float().cuda())  # ugly

            ans_scores = torch.sum(ans_word_scores, -1)
            ans_scores = ans_scores.view(batch_size, num_rounds, num_options)

            return ans_scores
