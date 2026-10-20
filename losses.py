# Margin-based classification headers that additionally return, for every sample,
#   CCS   = cos(x, W_y)              (similarity to its own class center)
#   NNCCS = max_{j != y} cos(x, W_j) (similarity to the nearest negative class center)
# whose ratio CCS / NNCCS is the per-epoch relative point margin (RPM) used by CARPM-FIQA.
# Adapted from CR-FIQA (https://github.com/fdbtrs/CR-FIQA).
import math

import torch
from torch import nn


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    return torch.div(input, norm)


@torch.no_grad()
def class_center_similarities(cos_theta, label):
    index = torch.where(label != -1)[0]
    ccs = cos_theta[index, label.view(-1)].detach().clone()
    negatives = cos_theta.detach().clone()
    negatives[index, label.view(-1)] = -1e-12
    nnccs, _ = negatives.max(dim=1)
    return ccs[index, None], nnccs[index, None]


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        ccs, nnccs = class_center_similarities(cos_theta, label)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta, ccs, nnccs


class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer("t", torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        ccs, nnccs = class_center_similarities(cos_theta, label)
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, ccs, nnccs
