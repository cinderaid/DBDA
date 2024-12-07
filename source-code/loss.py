import torch
import torch.nn.functional as F
# import ipdb
import torch.autograd as autograd
from torch.autograd import grad
from torch.autograd import Variable

from torch import nn


class Cross_entropy_loss(torch.nn.Module):
    def __init__(self):
        super(Cross_entropy_loss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred, labels):
        return self.criterion(pred, labels)


def gradient_discrepancy_loss_margin(loss_contra_s, loss_contra_t, net_g):
    gm_loss = 0
    grad_cossim11 = []

    for n, p in net_g.named_parameters():

        real_grad = grad([loss_contra_s],
                         [p],
                         create_graph=True,
                         only_inputs=True,
                         allow_unused=False)[0]
        fake_grad = grad([loss_contra_t],
                         [p],
                         create_graph=True,
                         only_inputs=True,
                         allow_unused=False)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        # _mse = F.mse_loss(fake_grad, real_grad)
        grad_cossim11.append(_cossim)
        # grad_mse.append(_mse)

    grad_cossim1 = torch.stack(grad_cossim11)
    gm_loss = (1.0 - grad_cossim1).mean()

    return gm_loss


class InvariancePenaltyLoss(nn.Module):
    r"""Invariance Penalty Loss from `Invariant Risk Minimization <https://arxiv.org/pdf/1907.02893.pdf>`_.
    We adopt implementation from `DomainBed <https://github.com/facebookresearch/DomainBed>`_. Given classifier
    output :math:`y` and ground truth :math:`labels`, we split :math:`y` into two parts :math:`y_1, y_2`, corresponding
    labels are :math:`labels_1, labels_2`. Next we calculate cross entropy loss with respect to a dummy classifier
    :math:`w`, resulting in :math:`grad_1, grad_2` . Invariance penalty is then :math:`grad_1*grad_2`.

    Inputs:
        - y: predictions from model
        - labels: ground truth

    Shape:
        - y: :math:`(N, C)` where C means the number of classes.
        - labels: :math:`(N, )` where N mean mini-batch size
    """

    def __init__(self):
        super(InvariancePenaltyLoss, self).__init__()
        self.scale = torch.tensor(1.).requires_grad_()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_1 = F.cross_entropy(y[::2] * self.scale, labels[::2])
        loss_2 = F.cross_entropy(y[1::2] * self.scale, labels[1::2])
        grad_1 = autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        return penalty