"""Apply APCI, SE-ARA and D-JOP to a shared relation classifier."""

import torch

from .intervention import intervene


def training_loss(classifier, batch, groups, risk, generator, *, apci=True, gate_penalty=0.01):
    """Compute a training loss and return the mask of changed image views.

    The classifier returns relation logits and gate weights, or None in place
    of weights for a classifier without the gate. The optimizer, checkpoint
    selection and evaluation remain with the caller.
    """
    clean_logits, gate = classifier(batch)
    identity = (gate - 1).square().mean(1) if gate is not None else None
    paired_logits, participates = None, None
    changed = torch.zeros_like(batch["gold"], dtype=torch.bool)
    if apci:
        paired_batch, changed, _ = intervene(batch, generator)
        paired_logits, paired_gate = classifier(paired_batch)
        participates = batch["modalities"].bool().any(1)
        if identity is not None:
            paired_identity = (paired_gate - 1).square().mean(1)
            identity = torch.where(participates, 0.5 * (identity + paired_identity), identity)

    loss = risk.objective(clean_logits, batch["gold"], groups, paired_logits, participates)
    if identity is not None:
        loss = loss + gate_penalty * identity.mean()
    return loss, changed
