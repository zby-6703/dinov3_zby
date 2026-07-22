from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from shipdraft_mtl.data.json_draft_e2e_dataset import _parse_draft_depth
from shipdraft_mtl.engine.config import Config, parse_override_options
from shipdraft_mtl.losses.draftformer_loss import build_loss
from shipdraft_mtl.losses.matcher import HungarianMatcher
from shipdraft_mtl.metrics import DraftFormerMetric
from shipdraft_mtl.modeling.backbones.dinov3_backbone import _load_dinov3_hub_model
from shipdraft_mtl.modeling.model import DraftFormerModel
from shipdraft_mtl.postprocess.draftformer_postprocess import DraftFormerPostProcess
from shipdraft_mtl.tools.train_multistage import STAGES, build_command
from shipdraft_mtl.utils import ckpt
from shipdraft_mtl.utils.task_train import apply_task_train_policy


def test_yaml_override_rejects_python_objects():
    with pytest.raises(yaml.constructor.ConstructorError):
        parse_override_options(["Global.value=!!python/object/apply:os.system ['echo unsafe']"])


def test_config_rejects_base_cycle(tmp_path: Path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text("_BASE_: second.yml\n", encoding="utf-8")
    second.write_text("_BASE_: first.yml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic"):
        Config(first)


def test_checkpoint_loader_uses_restricted_mode(monkeypatch):
    called = {}

    def fake_load(path, **kwargs):
        called.update(kwargs)
        return {"state_dict": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    ckpt._torch_load_checkpoint("model.pth", torch.device("cpu"))
    assert called["weights_only"] is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_draft_depth_is_not_trainable(value):
    depth, valid = _parse_draft_depth({"draft_depth": value})
    assert depth is None
    assert valid is False


def _postprocessor():
    return DraftFormerPostProcess(
        num_detection_queries=2,
        num_det_classes=1,
        keypoint_mode=True,
        waterline_score_thresh=0.3,
    )


def _postprocess_outputs(waterline_logit):
    return {
        "pred_logits": torch.tensor([[[0.0], [0.0], [waterline_logit], [waterline_logit]]]),
        "pred_points": torch.full((1, 4, 2), 0.5),
        "pred_boxes": torch.full((1, 4, 2), 0.5),
        "pred_masks": torch.zeros(1, 4, 1, 1),
    }


def test_waterline_threshold_is_not_bypassed():
    result = _postprocessor()(
        _postprocess_outputs(-2.0),
        [(16, 16)],
        [{"orig_height": 16, "orig_width": 16}],
    )[0]
    assert len(result["waterline_points"]) == 0
    assert result["sem_seg"].sum().item() == 0


def test_accepted_waterline_produces_binary_mask():
    result = _postprocessor()(
        _postprocess_outputs(0.0),
        [(16, 16)],
        [{"orig_height": 16, "orig_width": 16}],
    )[0]
    assert len(result["waterline_points"]) == 2
    assert result["sem_seg"].max().item() == 1.0


def _metric(**overrides):
    config = {
        "det_class_names": ["0"],
        "evaluate_character": False,
        "evaluate_waterline": False,
        "evaluate_depth": True,
        "evaluate_auxiliary": False,
        "main_indicator": "draft_score",
    }
    config.update(overrides)
    return DraftFormerMetric(config)


def test_no_valid_depth_sample_cannot_select_best_checkpoint():
    metric = _metric()
    metric(
        [{"draft_depth": torch.tensor(1.0), "draft_valid": torch.tensor(0.9)}],
        [{"draft_depth": torch.tensor(0.0), "draft_depth_valid": torch.tensor(False)}],
    )
    result = metric.get_metric()
    assert result["draft_score"] == 0.0
    assert result["selection_valid"] is False


def test_task_balancer_includes_active_depth_task():
    architecture = {
        "Head": {"num_classes": 1, "num_classes_seg": 1, "deep_supervision": False},
        "Decoder": {"num_detection_queries": 2, "dec_layers": 1},
        "DirectDepth": {"enabled": True},
        "TaskTrain": {"train_character": True, "train_waterline": True, "train_depth": True},
        "keypoint_mode": True,
    }
    loss = build_loss({"task_balancer": {"enabled": True}}, architecture)
    assert loss.task_balancer.task_names == ["det", "seg", "draft"]


def test_matcher_rejects_non_finite_costs():
    matcher = HungarianMatcher(cost_class=1.0, cost_box=1.0, cost_giou=0.0)
    outputs = {
        "pred_logits": torch.tensor([[[float("nan")]]]),
        "pred_boxes": torch.zeros(1, 1, 4),
    }
    targets = [{"labels": torch.tensor([0]), "boxes": torch.zeros(1, 4)}]
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        matcher(outputs, targets, cost=["cls", "box"])


def test_frozen_modules_stay_in_eval_mode():
    model = DraftFormerModel(
        backbone=nn.Dropout(),
        encoder=nn.Dropout(),
        decoder=nn.Dropout(),
        head=nn.Dropout(),
        loss=nn.Identity(),
        post_process=None,
        pixel_mean=(0.0, 0.0, 0.0),
        pixel_std=(1.0, 1.0, 1.0),
        direct_depth_head=nn.Dropout(),
    )
    cfg = {
        "TaskTrain": {
            "train_character": False,
            "train_waterline": False,
            "train_depth": True,
            "freeze": {"direct_depth_head": False},
        }
    }
    apply_task_train_policy(model, cfg)
    model.train()
    assert model.backbone.training is False
    assert model.encoder.training is False
    assert model.decoder.training is False
    assert model.head.training is False
    assert model.direct_depth_head.training is True


def test_multistage_no_eval_flag_is_forwarded():
    command = build_command("python", STAGES[0], None, [], False, True)
    assert "--no-eval" in command


def test_github_backbone_requires_explicit_remote_code_opt_in():
    with pytest.raises(ValueError, match="allow_remote_code"):
        _load_dinov3_hub_model(
            model_name="model",
            repo_or_dir="owner/repo",
            source="github",
            pretrained=True,
            weights=None,
            model_kwargs=None,
            force_reload=False,
            trust_repo=False,
            skip_validation=False,
            verbose=False,
            allow_remote_code=False,
        )
