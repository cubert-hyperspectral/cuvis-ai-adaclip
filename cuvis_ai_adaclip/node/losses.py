"""Loss nodes for AdaCLIP training-path integration."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class LossNode(Node):
    """Base class for loss nodes restricted to train/val/test stages."""

    def __init__(self, **kwargs) -> None:
        assert "execution_stages" not in kwargs, (
            "Loss nodes can only execute in train, val, and test stages."
        )
        super().__init__(
            execution_stages={
                ExecutionStage.TRAIN,
                ExecutionStage.VAL,
                ExecutionStage.TEST,
            },
            **kwargs,
        )


class AdaCLIPFocalDiceLoss(LossNode):
    """Combined Focal + Dice loss for AdaCLIP training.

    Supports two execution paths:
    - per-layer path using `per_layer_scores` and optional `image_score_2ch`
    - fallback path using aggregated `predictions`
    """

    INPUT_SPECS = {
        "predictions": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Aggregated anomaly scores [B, H, W, 1] for fallback path",
        ),
        "targets": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Ground truth binary masks [B, H, W, 1]",
        ),
        "per_layer_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-layer softmaxed maps [B, num_layers*2, H, W]",
            optional=True,
        ),
        "image_score_2ch": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Image-level score [B, 2] in [normal, anomaly] order",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Combined focal + dice loss")
    }

    def __init__(
        self,
        weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_smooth: float = 1e-5,
        focal_alpha: float | list[float] | None = None,
        focal_balance_index: int = 1,
        image_loss_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.weight = weight
        self.focal_gamma = focal_gamma
        self.focal_smooth = focal_smooth
        self.focal_alpha = focal_alpha
        self.focal_balance_index = focal_balance_index
        self.image_loss_weight = image_loss_weight

        super().__init__(
            weight=weight,
            focal_gamma=focal_gamma,
            focal_smooth=focal_smooth,
            focal_alpha=focal_alpha,
            focal_balance_index=focal_balance_index,
            image_loss_weight=image_loss_weight,
            **kwargs,
        )

    def _resolve_alpha(self, num_class: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        alpha = self.focal_alpha
        if alpha is None:
            return torch.ones(num_class, 1, device=device, dtype=dtype)
        if isinstance(alpha, float):
            alpha_vec = torch.ones(num_class, 1, device=device, dtype=dtype) * (1.0 - alpha)
            alpha_vec[self.focal_balance_index] = alpha
            return alpha_vec
        if isinstance(alpha, list):
            if len(alpha) != num_class:
                raise ValueError(
                    f"focal_alpha list length must equal num_class ({num_class}), got {len(alpha)}"
                )
            alpha_vec = torch.tensor(alpha, device=device, dtype=dtype).view(num_class, 1)
            alpha_sum = alpha_vec.sum().clamp_min(self.focal_smooth)
            return alpha_vec / alpha_sum
        raise TypeError("focal_alpha must be None, float, or list[float]")

    def _mc_focal(self, logit: Tensor, target: Tensor) -> Tensor:
        num_class = logit.shape[1]
        gamma = self.focal_gamma
        smooth = self.focal_smooth

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
            target = target.reshape(-1, 1).long()

        alpha = self._resolve_alpha(num_class, logit.device, logit.dtype)

        one_hot = torch.zeros(target.size(0), num_class, device=logit.device, dtype=logit.dtype)
        one_hot.scatter_(1, target, 1)

        if smooth:
            one_hot = one_hot.clamp(smooth / max(1, num_class - 1), 1.0 - smooth)

        pt = (one_hot * logit).sum(1).clamp_min(smooth)
        logpt = pt.log()

        alpha = alpha[target.squeeze(1)].squeeze()
        loss = -1.0 * alpha * (1 - pt).pow(gamma) * logpt
        return loss.mean()

    @staticmethod
    def _dice(pred: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
        n = pred.shape[0]
        pred_flat = pred.reshape(n, -1)
        tgt_flat = target.float().reshape(n, -1)
        intersection = (pred_flat * tgt_flat).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) + smooth)
        return 1.0 - dice.sum() / n

    @staticmethod
    def _binary_focal(
        pred: Tensor, target: Tensor, gamma: float = 2.0, smooth: float = 1e-5
    ) -> Tensor:
        pred = pred.clamp(smooth, 1.0 - smooth)
        target = target.float()
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt).pow(gamma)
        bce = -(target * pred.log() + (1 - target) * (1 - pred).log())
        return (focal_weight * bce).mean()

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        per_layer_scores: Tensor | None = None,
        image_score_2ch: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        gt = targets.float().squeeze(-1)
        gt_binary = (gt > 0.5).float()

        has_per_layer = (
            per_layer_scores is not None
            and per_layer_scores.dim() == 4
            and per_layer_scores.shape[1] > 1
        )

        if has_per_layer:
            n_layers = per_layer_scores.shape[1] // 2
            seg_loss = torch.tensor(
                0.0, device=per_layer_scores.device, dtype=per_layer_scores.dtype
            )

            for i in range(n_layers):
                am = per_layer_scores[:, i * 2 : (i + 1) * 2, :, :]
                seg_loss = seg_loss + self._mc_focal(am, gt_binary)
                seg_loss = seg_loss + self._dice(am[:, 1, :, :], gt_binary)
                seg_loss = seg_loss + self._dice(am[:, 0, :, :], 1.0 - gt_binary)

            loss = seg_loss

            if (
                image_score_2ch is not None
                and image_score_2ch.dim() == 2
                and image_score_2ch.shape[1] == 2
            ):
                is_anomaly = (gt_binary.flatten(1).sum(dim=1) > 0).long()
                cls_loss = self._mc_focal(image_score_2ch.unsqueeze(-1), is_anomaly.unsqueeze(-1))
                loss = loss + self.image_loss_weight * cls_loss
        else:
            pred = predictions.squeeze(-1)
            # Fallback contract:
            # - if values are outside [0,1], treat as logits and apply sigmoid
            # - otherwise treat as probabilities and clamp safely
            if pred.min() < 0.0 or pred.max() > 1.0:
                pred = torch.sigmoid(pred)
            else:
                pred = pred.clamp(self.focal_smooth, 1.0 - self.focal_smooth)

            loss = self._binary_focal(pred, gt_binary, self.focal_gamma, self.focal_smooth)
            loss = loss + self._dice(pred, gt_binary) + self._dice(1.0 - pred, 1.0 - gt_binary)

        return {"loss": self.weight * loss}


__all__ = ["LossNode", "AdaCLIPFocalDiceLoss"]
