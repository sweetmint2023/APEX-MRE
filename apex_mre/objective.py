"""Clean group-risk updates and the joint classification objective."""

import torch
from torch.nn import functional as F


def js_rows(clean_logits, paired_logits):
    """Jensen–Shannon divergence in nats for each pair of predictions."""
    clean_log = clean_logits.log_softmax(-1)
    paired_log = paired_logits.log_softmax(-1)
    clean, paired = clean_log.exp(), paired_log.exp()
    mixture_log = (0.5 * (clean + paired)).clamp_min(torch.finfo(clean.dtype).tiny).log()
    return 0.5 * ((clean * (clean_log - mixture_log)).sum(-1)
                  + (paired * (paired_log - mixture_log)).sum(-1))


class ExposureRisk:
    """Self-normalized exposure risk with per-epoch group updates."""

    def __init__(self, counts, device="cpu", *, beta=0.5, consistency=0.5, eta=0.2, coupled=False):
        self.counts = torch.as_tensor(counts, dtype=torch.float64, device=device)
        if self.counts.ndim != 1 or len(self.counts) == 0 or not (self.counts > 0).all():
            raise ValueError("Each retained training group needs a positive count")
        self.prevalence = self.counts / self.counts.sum()
        self.q = torch.ones_like(self.counts) / len(self.counts)
        self.beta, self.consistency, self.eta = beta, consistency, eta
        self.coupled = coupled
        self.start_epoch()

    def start_epoch(self):
        self.loss_sum = torch.zeros_like(self.counts)
        self.seen = torch.zeros_like(self.counts)

    def objective(self, clean_logits, labels, group_ids, paired_logits=None, participates=None):
        clean_ce = F.cross_entropy(clean_logits, labels, reduction="none")
        clean_norm = clean_ce
        paired_ce = torch.zeros_like(clean_ce)
        consistency = torch.zeros_like(clean_ce)
        if paired_logits is not None:
            if participates is None or participates.shape != labels.shape or participates.dtype != torch.bool:
                raise ValueError("Supply one Boolean participation flag per example")
            mask = participates.to(clean_ce.dtype)
            clean_norm = (1 - 0.5 * mask) * clean_ce
            paired_ce = 0.5 * mask * F.cross_entropy(paired_logits, labels, reduction="none")
            consistency = mask * js_rows(clean_logits, paired_logits)

        weights = (self.q[group_ids] / self.prevalence[group_ids]).to(clean_ce.dtype)
        if self.coupled:
            signal = clean_norm + paired_ce + self.consistency * consistency
            loss = (1 - self.beta) * signal.mean() + self.beta * (weights * signal).sum() / weights.sum()
        else:
            signal = clean_ce
            group_risk = (weights * clean_norm).sum() / weights.sum()
            loss = ((1 - self.beta) * clean_norm.mean() + self.beta * group_risk
                    + paired_ce.mean() + self.consistency * consistency.mean())

        self.loss_sum.index_add_(0, group_ids, signal.detach().to(torch.float64))
        self.seen.index_add_(0, group_ids, torch.ones_like(clean_ce, dtype=torch.float64))
        return loss

    def finish_epoch(self):
        if not torch.equal(self.seen, self.counts):
            raise RuntimeError("Epoch group counts must cover the complete training split")
        group_mean = self.loss_sum / self.seen
        if self.beta > 0 and self.eta > 0:
            self.q *= torch.exp(self.eta * (group_mean - group_mean.max()))
            self.q /= self.q.sum()
        return group_mean
