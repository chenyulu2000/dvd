import torch
import torch.nn as nn
import torch.nn.functional as f


def ndcg_loss(output, labels, log_softmax=nn.LogSoftmax(dim=-1)):
    output = log_softmax(output)
    batch_size, num_options = output.size()
    labels = labels.view(batch_size, -1)
    output = output.view(batch_size, -1)
    loss = -torch.mean(torch.sum(labels * output, dim=1))
    return loss


def ce_loss(decoder_type, batch, output):
    if decoder_type == 'disc':
        target = batch['ans_ind']
        criterion = nn.CrossEntropyLoss()
    else:
        target = batch['ans_out']
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    return loss


def generalized_ce_loss(decoder_type, batch, output, q):
    class GeneralizedCrossEntropyLoss(nn.Module):
        def __init__(self, cross_entropy, q=0.7):
            super(GeneralizedCrossEntropyLoss, self).__init__()
            self.q = q
            self.cross_entropy_loss = cross_entropy

        def forward(self, output, target):
            p = f.softmax(output, dim=1)
            Yg = torch.gather(p, 1, target.view(-1, 1))
            loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
            loss = self.cross_entropy_loss(output, target) * loss_weight
            return loss.mean()

    if decoder_type == 'disc':
        target = batch['ans_ind']
        criterion = GeneralizedCrossEntropyLoss(cross_entropy=nn.CrossEntropyLoss(reduction='none'), q=q)
    else:
        target = batch['ans_out']
        criterion = GeneralizedCrossEntropyLoss(cross_entropy=nn.CrossEntropyLoss(
            ignore_index=0, reduction='none'), q=q)
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    return loss
