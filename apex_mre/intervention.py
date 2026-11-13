"""Replace context views while retaining the supplied relation arguments."""

import torch


def protected_context_slots(batch, atol=1e-6):
    """Identify real view slots that coincide with a visual endpoint box."""
    boxes = batch["boxes"]
    targets = batch["target_boxes"]
    visual = batch["modalities"].bool()
    difference = (boxes[:, :, None, :] - targets[:, None, :, :]).abs()
    matches = (difference <= atol).all(-1) & visual[:, None, :]
    return matches.any(-1) & batch["real"].bool()


def intervene(batch, generator):
    """Draw cyclic donors within each directed modality and slot.

    Use a CPU generator. Reject donors from the same RGB source and keep the
    original view in those slots. The returned batch shares all non-view fields
    with the input, including the separate visual argument crops.
    """
    protected = protected_context_slots(batch)
    modalities = batch["modalities"].detach().cpu().long()
    codes = modalities[:, 0] * 2 + modalities[:, 1]
    visual = codes.ne(0)
    eligible = (batch["real"].bool() & ~protected).detach().cpu()
    sources = batch["source_code"].detach().cpu()
    donors = torch.full(eligible.shape, -1, dtype=torch.long)
    replacements = batch["box_imgs"].clone()
    rejected = 0
    for code in (1, 2, 3):
        for slot in range(eligible.shape[1]):
            rows = torch.nonzero(codes.eq(code) & eligible[:, slot], as_tuple=True)[0]
            if len(rows) < 2:
                continue
            recipients = rows[torch.randperm(len(rows), generator=generator)]
            candidates = recipients.roll(1)
            different_source = sources[recipients].ne(sources[candidates])
            rejected += int((~different_source).sum())
            recipients, candidates = recipients[different_source], candidates[different_source]
            donors[recipients, slot] = candidates
            replacements[recipients.to(replacements.device), slot] = batch["box_imgs"][
                candidates.to(replacements.device), slot
            ]

    changed_slots = replacements.ne(batch["box_imgs"]).flatten(2).any(2)
    if changed_slots[protected].any() or changed_slots[~visual.to(protected.device)].any():
        raise RuntimeError("Intervention changed a protected view or a TT row")

    return dict(batch, box_imgs=replacements), changed_slots.any(1), {
        "donors": donors,
        "protected_slots": int(protected.sum()),
        "same_source_rejected_slots": rejected,
        "pixel_changed_slots": int(changed_slots.sum()),
    }
