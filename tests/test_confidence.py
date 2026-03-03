from src.strategies.fast_text import FastTextExtractor


def test_confidence_high_for_good_text_signal():
    ex = FastTextExtractor(thresholds={
        "target_chars_per_page": 350,
        "target_density": 0.001,
        "max_images_for_ratio": 10,
    })
    conf = ex.score_confidence(
        char_counts=[600, 550],
        densities=[0.0012, 0.0011],
        image_ratios=[0.05, 0.1],
    )
    assert conf >= 0.8


def test_confidence_low_for_scanned_like_signal():
    ex = FastTextExtractor(thresholds={
        "target_chars_per_page": 350,
        "target_density": 0.001,
        "max_images_for_ratio": 10,
    })
    conf = ex.score_confidence(
        char_counts=[20, 10],
        densities=[0.00001, 0.00001],
        image_ratios=[0.9, 0.95],
    )
    assert conf < 0.4
