from pathlib import Path

import pytest

from src.config import load_config


def test_load_config_applies_defaults(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
triage:
  thresholds:
    low_density_threshold: 0.0002
    high_density_threshold: 0.001
    image_heavy_threshold: 0.6
    max_images_for_ratio: 10
  domain_keywords:
    financial: ["revenue"]
extraction:
  confidence_minimum: 0.7
  budget_per_document_usd: 0.3
chunking:
  max_tokens: 400
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["extraction"]["enforce_hard_caps"] is True
    assert cfg["extraction"]["vlm_budget"]["max_pages_per_document"] == 25
    assert cfg["extraction"]["layout"]["adapter"]["provider"] == "heuristic"
    assert cfg["extraction"]["escalation"]["chains"]["fast_text"] == ["fast_text", "layout_aware", "vision_augmented"]


def test_load_config_rejects_invalid_confidence(tmp_path: Path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        """
triage:
  thresholds:
    low_density_threshold: 0.0002
    high_density_threshold: 0.001
    image_heavy_threshold: 0.6
    max_images_for_ratio: 10
  domain_keywords:
    financial: ["revenue"]
extraction:
  confidence_minimum: 2.0
  budget_per_document_usd: 0.3
chunking:
  max_tokens: 400
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(cfg_path)
