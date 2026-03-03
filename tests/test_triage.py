from src.agents.triage import TriageAgent, PageStat
from src.models import OriginType


def test_origin_type_scanned():
    agent = TriageAgent(domain_keywords={"financial": ["revenue"]}, thresholds={
        "low_density_threshold": 0.0002,
        "high_density_threshold": 0.001,
        "image_heavy_threshold": 0.6,
        "max_images_for_ratio": 10,
    })
    origin = agent._origin_type(avg_density=0.00005, avg_image_ratio=0.8)
    assert origin == OriginType.SCANNED_IMAGE


def test_domain_hint_financial():
    agent = TriageAgent(
        domain_keywords={"financial": ["revenue", "balance sheet"], "legal": ["clause"]},
        thresholds={
            "low_density_threshold": 0.0002,
            "high_density_threshold": 0.001,
            "image_heavy_threshold": 0.6,
            "max_images_for_ratio": 10,
        },
    )
    assert agent._domain_hint("The revenue and balance sheet were audited") == "financial"
