"""Visual endpoint injection and seven-view residual gating for FocalMRE."""

import torch
from torch import nn
from torch.nn import functional as F


class PairResidualGate(nn.Module):
    """Score each visual view given the ordered head and tail representations."""

    def __init__(self, hidden=128):
        super().__init__()
        self.pair_project = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, hidden), nn.GELU())
        self.view_project = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, hidden), nn.GELU())
        self.view_type = nn.Embedding(2, hidden)
        self.score = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.register_buffer("types", torch.tensor([0, 0, 0, 0, 1, 1, 1]))

    def forward(self, pair, residues):
        query = self.pair_project(pair).unsqueeze(1).expand(-1, 7, -1)
        views = self.view_project(residues) + self.view_type(self.types).unsqueeze(0)
        scores = self.score(torch.cat((query, views, query * views), dim=-1)).squeeze(-1)
        return 2 * torch.sigmoid(scores), scores


class UnifiedFocal(nn.Module):
    """Wrap an initialized BERT/CLIP FocalMRE classifier for given endpoints."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        device = next(base.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self.gate = PairResidualGate().to(device)
            self.target_norm = nn.LayerNorm(768).to(device)
            self.target_project = nn.Linear(768, 768, bias=False).to(device)
            nn.init.eye_(self.target_project.weight)
            self.target_gain = nn.Parameter(torch.tensor(0.1, device=device))

        self.context = None
        self.captured = None
        self.base.model.text_model.register_forward_hook(self._inject_targets)
        fuse_view = self.base.model.fuse_img_feat

        def capture(text, index, image):
            fused = fuse_view(text, index, image)
            if self.captured is not None:
                self.captured["views"][index] = fused.transpose(0, 1)
            return fused

        self.base.model.fuse_img_feat = capture

    def _inject_targets(self, _module, _inputs, output):
        if self.context is None:
            raise RuntimeError("Call the endpoint adapter to run the backbone")
        text = output.last_hidden_state
        targets, visual, positions = self.context
        if visual.any():
            text = text.clone()
            for side in (0, 1):
                rows = torch.nonzero(visual[:, side], as_tuple=True)[0]
                markers = positions[rows, side]
                text[rows, markers] = text[rows, markers] + self.target_gain * targets[rows, side]
            output.last_hidden_state = text
        if self.captured is not None:
            self.captured["text"] = text
        return output

    def forward(self, batch, mode="gate"):
        if mode not in ("control", "gate"):
            raise ValueError("Expected control or gate mode")
        ids = batch["input_ids"]
        markers = (ids.eq(self.base.head_start), ids.eq(self.base.tail_start))
        if any(not (mask.sum(1) == 1).all() for mask in markers):
            raise ValueError("Each sample needs one head marker and one tail marker")
        positions = torch.stack([mask.int().argmax(1) for mask in markers], dim=1)
        visual = batch["modalities"].bool()
        targets = self.target_gain.new_zeros((len(ids), 2, 768))
        if visual.any():
            encoded = self.base.model.vision_model(pixel_values=batch["target_imgs"][visual], return_dict=True)
            features = encoded.last_hidden_state[:, 1:].mean(1)
            targets[visual] = self.target_project(self.target_norm(features))

        self.context = (targets, visual, positions)
        self.captured = {"views": [None] * 7} if mode == "gate" else None
        try:
            inputs = {key: batch[key] for key in ("input_ids", "token_type_ids", "attention_mask", "box_imgs")}
            original = self.base(**inputs)
            if mode == "control":
                return original, None

            rows = torch.arange(len(ids), device=ids.device)

            def endpoint_pair(states):
                return states[rows[:, None], positions].flatten(1)

            pair = endpoint_pair(self.captured["text"])
            residues = torch.stack([endpoint_pair(view) - pair for view in self.captured["views"]], dim=1)
            weights, _ = self.gate(pair, residues)
            fusion_weights = torch.cat((self.base.model.weights1.flatten(), self.base.model.weights2.flatten()))
            first, second = self.base.classifier.weight.split(pair.shape[-1], dim=1)
            contributions = torch.stack([
                F.linear(residues[:, index], first if index < 4 else second) * fusion_weights[index]
                for index in range(7)
            ], dim=1)
            logits = original + ((weights - 1).unsqueeze(-1) * contributions).sum(1)
            return logits, weights
        finally:
            self.context = None
            self.captured = None
