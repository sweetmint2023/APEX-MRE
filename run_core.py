"""Check the core method on a small batch of synthetic tensors."""

import torch
from torch.nn import functional as F

from apex_mre.classifier import PairResidualGate
from apex_mre.exposure import ExposureGroups
from apex_mre.intervention import intervene, protected_context_slots
from apex_mre.objective import ExposureRisk


def make_batch():
    modalities = torch.tensor([[0, 0]] * 2 + [[0, 1]] * 4 + [[1, 0]] * 2 + [[1, 1]] * 4)
    source_codes = torch.tensor([0, 1, 0, 0, 0, 0, 4, 5, 6, 6, 7, 8])
    rows = []
    for i, (head, tail) in enumerate(modalities.tolist()):
        rows.append({
            "img_id": f"example_{i}",
            "h": {"name": f"head_{i % 3}", "pos": [0, 0, 1, 1] if head else [0, 1]},
            "t": {"name": f"tail_{i % 2}", "pos": [0, 0, 1, 1] if tail else [1, 2]},
        })
    groups = ExposureGroups(rows, {row["img_id"]: s for row, s in zip(rows, source_codes.tolist())})
    n = len(rows)
    boxes = torch.tensor([[0, 0, .3, .3], [.3, .3, .7, .7], [.7, .7, 1, 1]] + [[0, 0, 0, 0]] * 4)
    boxes = boxes.expand(n, -1, -1)
    real = torch.tensor([[True] * 3 + [False] * 4]).expand(n, -1)
    views = torch.rand(n, 7, 3, 4, 4) * real[:, :, None, None, None]
    batch = {
        "input_ids": torch.randint(16, (n, 4)),
        "box_imgs": views,
        "boxes": boxes,
        "target_boxes": boxes[:, :2],
        "target_imgs": views[:, :2] * modalities[:, :, None, None, None],
        "real": real,
        "modalities": modalities,
        "source_code": source_codes,
        "gold": torch.arange(n) % 3,
        "row": torch.arange(n),
    }
    return batch, groups


def main():
    torch.set_num_threads(1)
    generator = torch.Generator()
    batch, groups = make_batch()
    paired, changed, stats = intervene(batch, generator)
    protected = protected_context_slots(batch)
    torch.testing.assert_close(batch["box_imgs"][protected], paired["box_imgs"][protected])
    assert paired["target_imgs"] is batch["target_imgs"]
    assert paired["input_ids"] is batch["input_ids"]
    assert changed.any() and not changed[:2].any()
    assert stats["same_source_rejected_slots"] > 0

    n = len(batch["gold"])
    weights, _ = PairResidualGate()(torch.rand(n, 1536), torch.rand(n, 7, 1536))
    torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0, atol=0)

    logits = torch.randn(n, 3, requires_grad=True)
    paired_logits = torch.randn(n, 3, requires_grad=True)
    participates = batch["modalities"].bool().any(1)
    risk = ExposureRisk(groups.counts)
    loss = risk.objective(logits, batch["gold"], groups.train_ids, paired_logits, participates)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(paired_logits.grad).all()
    assert not paired_logits.grad[~participates].any()

    clean_ce = F.cross_entropy(logits.detach(), batch["gold"], reduction="none").double()
    expected = torch.stack([clean_ce[groups.train_ids == group].mean() for group in range(len(groups.groups))])
    torch.testing.assert_close(risk.finish_epoch(), expected)
    torch.testing.assert_close(risk.q, torch.softmax(risk.eta * expected, dim=0))
    print("APCI, SE-ARA, D-JOP, and gate initialization checks passed.")


if __name__ == "__main__":
    main()
