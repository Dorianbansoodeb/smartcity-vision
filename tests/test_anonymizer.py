"""Behaviour of the privacy anonymiser.

The contract is that identifiable pixels inside a region are actually changed,
not that a Haar cascade finds a face in a unit test. Cascades are too brittle
for that; the test injects a known box instead.
"""

from __future__ import annotations

import numpy as np

from smartcity_vision.detection.detector import Detection
from smartcity_vision.privacy.anonymizer import FrameAnonymizer
from smartcity_vision.utils.config import PrivacyConfig


def face_like_frame() -> np.ndarray:
    """A high-contrast patterned square on a black background.

    A uniform colour would be unchanged by a Gaussian blur, so the region has
    to contain spatial detail for the test to be meaningful.
    """
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:30, 0:30]
    image[20:50, 20:50] = np.stack([((xx * 7 + yy * 11) % 256).astype(np.uint8)] * 3, axis=-1)
    return image


def test_blur_changes_pixels_inside_the_region_and_not_outside() -> None:
    anonymizer = FrameAnonymizer(PrivacyConfig(enabled=True, method="blur", blur_strength=21))
    original = face_like_frame()

    redacted = anonymizer.redact_regions(original, [(20, 20, 30, 30)])

    interior = np.s_[25:45, 25:45]
    exterior = np.s_[0:10, 0:10]
    assert not np.array_equal(redacted[interior], original[interior])
    assert np.array_equal(redacted[exterior], original[exterior])
    assert np.array_equal(original[interior], face_like_frame()[interior]), (
        "input must be unchanged"
    )


def test_pixelate_reduces_unique_colours_inside_the_region() -> None:
    anonymizer = FrameAnonymizer(PrivacyConfig(enabled=True, method="pixelate", pixel_size=10))
    original = face_like_frame()
    original[20:50, 20:50] = np.random.default_rng(0).integers(
        0, 255, size=(30, 30, 3), dtype=np.uint8
    )

    redacted = anonymizer.redact_regions(original, [(20, 20, 30, 30)])

    unique_original = len(np.unique(original[20:50, 20:50].reshape(-1, 3), axis=0))
    unique_redacted = len(np.unique(redacted[20:50, 20:50].reshape(-1, 3), axis=0))
    assert unique_redacted < unique_original


def test_detection_fallback_redacts_a_person_head_region() -> None:
    anonymizer = FrameAnonymizer(PrivacyConfig(enabled=True, method="blur", blur_strength=15))
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:100, 0:100]
    image[:] = np.stack([((xx + yy) % 256).astype(np.uint8)] * 3, axis=-1)
    person = Detection(
        class_id=0, class_name="person", confidence=0.9, bbox=(10.0, 10.0, 50.0, 90.0), track_id=1
    )

    redacted, stats = anonymizer.anonymize(image, [person])

    assert stats.faces >= 1
    assert not np.array_equal(redacted[12:30, 15:45], image[12:30, 15:45])
    assert np.array_equal(redacted[0:8, 0:8], image[0:8, 0:8])


def test_disabled_anonymiser_returns_the_same_array() -> None:
    anonymizer = FrameAnonymizer(PrivacyConfig(enabled=False))
    image = face_like_frame()

    output, stats = anonymizer.anonymize(image)

    assert output is image
    assert stats.total == 0
