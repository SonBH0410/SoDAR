"""Macro-per-image and micro-pixel binary scores, plus eight category strata."""

from collections import defaultdict

import torch


class SegmentationMeter:
    def __init__(self):
        self.by_case = []
        self.tp = self.fp = self.fn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, truth: torch.Tensor, labels, sample_ids):
        pred = torch.sigmoid(logits) >= 0.5
        truth = truth >= 0.5
        if pred.shape != truth.shape:
            raise ValueError(f"Shapes differ: {pred.shape} vs {truth.shape}")
        for p, y, category, sample_id in zip(pred, truth, labels, sample_ids):
            tp = int((p & y).sum().item())
            fp = int((p & ~y).sum().item())
            fn = int((~p & y).sum().item())
            self.tp += tp
            self.fp += fp
            self.fn += fn
            # Both empty = 1; no match with foreground = 0.
            dice = (2*tp) / (2*tp + fp + fn) if 2*tp + fp + fn else 1.0
            iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
            self.by_case.append({"id": sample_id, "label": int(category),
                                 "dice": dice, "iou": iou, "tp": tp, "fp": fp, "fn": fn})

    def compute(self) -> dict:
        if not self.by_case:
            raise ValueError("Cannot evaluate empty loader")
        mean = lambda key, rows: sum(r[key] for r in rows) / len(rows)
        grouped = defaultdict(list)
        for row in self.by_case:
            grouped[row["label"]].append(row)
        return {
            "n_images": len(self.by_case),
            "macro_dice": mean("dice", self.by_case),
            "macro_iou": mean("iou", self.by_case),
            "micro_dice": 2*self.tp/(2*self.tp+self.fp+self.fn) if 2*self.tp+self.fp+self.fn else 1.0,
            "micro_iou": self.tp/(self.tp+self.fp+self.fn) if self.tp+self.fp+self.fn else 1.0,
            "classwise": {str(c): {"count": len(rows), "dice": mean("dice", rows),
                                    "iou": mean("iou", rows)} for c, rows in sorted(grouped.items())},
        }
