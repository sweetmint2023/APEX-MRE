"""Fit exposure groups from the training split and identify RGB sources."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch


def textual_names(row):
    for side in ("h", "t"):
        endpoint = row[side]
        if len(endpoint["pos"]) == 2:
            name = endpoint["name"]
            if isinstance(name, list):
                name = " ".join(name)
            yield " ".join(name.casefold().split())


class ExposureGroups:
    """Group by directed modality, minimum name frequency, and image reuse."""

    def __init__(self, train_rows, source_by_image, reuse_threshold=4):
        self.source_by_image = source_by_image
        self.reuse_threshold = reuse_threshold
        self.name_counts = Counter(name for row in train_rows for name in textual_names(row))
        self.source_counts = Counter(self.source(row) for row in train_rows)
        keys = [self.key(row) for row in train_rows]
        self.groups = sorted(set(keys))
        id_by_key = {key: i for i, key in enumerate(self.groups)}
        self.train_ids = torch.tensor([id_by_key[key] for key in keys], dtype=torch.long)
        self.counts = torch.bincount(self.train_ids, minlength=len(self.groups))

    def source(self, row):
        return self.source_by_image[row["img_id"]]

    def key(self, row):
        modality = "".join("V" if len(row[side]["pos"]) == 4 else "T" for side in ("h", "t"))
        names = list(textual_names(row))
        if names:
            frequency = min(self.name_counts[name] for name in names)
            bucket = "low" if frequency <= 2 else "medium" if frequency <= 10 else "high"
        else:
            bucket = "no_text_endpoint"
        return modality, bucket, self.source_counts[self.source(row)] > self.reuse_threshold


def read_rows(path):
    """Read the JSON array or line-delimited records from benchmark releases."""
    content = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [ast.literal_eval(line) for line in content.splitlines() if line.strip()]


def rgb_sources(rows, image_dir):
    """Identify matching decoded RGB images even when file names differ."""
    sources = {}
    for image_id in sorted({row["img_id"] for row in rows}):
        with Image.open(Path(image_dir) / image_id) as image:
            rgb = image.convert("RGB")
            header = ("%d,%d:" % rgb.size).encode("ascii")
            sources[image_id] = hashlib.sha256(header + rgb.tobytes()).hexdigest()
    return sources


def main():
    parser = argparse.ArgumentParser(description="Compute exposure groups on a benchmark training split.")
    parser.add_argument("--train", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir", type=Path, help="Directory containing the released images.")
    source.add_argument("--source-index", type=Path, help="Existing image-to-source JSON index.")
    args = parser.parse_args()
    rows = read_rows(args.train)
    if args.source_index:
        index = json.loads(args.source_index.read_text(encoding="utf-8"))
        sources = {key: value["pixel_sha256"] if isinstance(value, dict) else value
                   for key, value in index.items()}
    else:
        sources = rgb_sources(rows, args.image_dir)
    exposure = ExposureGroups(rows, sources)
    groups = [
        {"modality": modality, "name_exposure": bucket, "high_source_reuse": reuse, "count": count}
        for (modality, bucket, reuse), count in zip(exposure.groups, exposure.counts.tolist())
    ]
    print(json.dumps({"train_rows": len(rows), "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
