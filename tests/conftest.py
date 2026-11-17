"""Shared fixtures for chromatone tests."""

from __future__ import annotations

import pytest

from chromatone import SPD, get_illuminant, rgb_led_set


@pytest.fixture
def d65() -> SPD:
    return get_illuminant("D65")


@pytest.fixture
def illuminant_a() -> SPD:
    return get_illuminant("A")


@pytest.fixture
def equal_energy() -> SPD:
    return get_illuminant("E")


@pytest.fixture
def grey_reflectance() -> SPD:
    return SPD.ones(name="grey").scale(0.5)


@pytest.fixture
def led_channels():
    return rgb_led_set(fwhm_nm=25.0)
