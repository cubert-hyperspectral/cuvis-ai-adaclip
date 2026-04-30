"""Tests for plugin-native AdaCLIP loss nodes."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_adaclip import AdaCLIPFocalDiceLoss


class TestAdaCLIPFocalDiceLoss:
    def test_forward_with_per_layer_scores_is_scalar_and_differentiable(self) -> None:
        node = AdaCLIPFocalDiceLoss(weight=1.0, focal_gamma=2.0, image_loss_weight=1.0)
        b, h, w = 2, 16, 16

        raw = torch.randn(b, 4, h, w, requires_grad=True)
        pl = torch.softmax(raw.reshape(b, 2, 2, h, w), dim=2).reshape(b, 4, h, w)

        image_raw = torch.randn(b, 2, requires_grad=True)
        image_score_2ch = torch.softmax(image_raw, dim=1)

        predictions = torch.rand(b, h, w, 1, requires_grad=True)
        targets = torch.randint(0, 2, (b, h, w, 1)).bool()

        out = node.forward(
            predictions=predictions,
            targets=targets,
            per_layer_scores=pl,
            image_score_2ch=image_score_2ch,
        )
        loss = out["loss"]
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad

        loss.backward()
        assert raw.grad is not None

    def test_forward_fallback_accepts_logits(self) -> None:
        node = AdaCLIPFocalDiceLoss(weight=1.0, focal_gamma=2.0, image_loss_weight=1.0)
        b, h, w = 2, 16, 16
        predictions = torch.randn(b, h, w, 1, requires_grad=True)  # logits, unconstrained range
        targets = torch.randint(0, 2, (b, h, w, 1)).bool()

        out = node.forward(predictions=predictions, targets=targets)
        loss = out["loss"]
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad

        loss.backward()
        assert predictions.grad is not None

    def test_focal_alpha_none_is_backward_compatible(self) -> None:
        b, h, w = 1, 8, 8
        predictions = torch.rand(b, h, w, 1)
        targets = torch.randint(0, 2, (b, h, w, 1)).bool()

        node = AdaCLIPFocalDiceLoss(focal_alpha=None)
        loss = node.forward(predictions=predictions, targets=targets)["loss"]
        assert torch.isfinite(loss)

    def test_focal_alpha_list_supported(self) -> None:
        b, h, w = 1, 8, 8
        predictions = torch.rand(b, h, w, 1)
        targets = torch.randint(0, 2, (b, h, w, 1)).bool()

        node = AdaCLIPFocalDiceLoss(focal_alpha=[0.25, 0.75])
        loss = node.forward(predictions=predictions, targets=targets)["loss"]
        assert torch.isfinite(loss)

    def test_invalid_focal_alpha_list_length_raises(self) -> None:
        node = AdaCLIPFocalDiceLoss(focal_alpha=[1.0, 1.0, 1.0])
        b, h, w = 1, 8, 8
        predictions = torch.rand(b, h, w, 1)
        targets = torch.randint(0, 2, (b, h, w, 1)).bool()
        per_layer_scores = torch.softmax(torch.rand(b, 4, h, w), dim=1)

        with pytest.raises(ValueError):
            node.forward(
                predictions=predictions,
                targets=targets,
                per_layer_scores=per_layer_scores,
                image_score_2ch=torch.softmax(torch.rand(b, 2), dim=1),
            )
