"""chromatone — spectral colour science toolkit.

Stdlib-only library for spectral power distributions (SPD), CIE XYZ/Lab
conversions, standard illuminants, chromatic adaptation, colour
differences, and simple LED/SPD mixing.

Quick start
-----------
>>> from chromatone import get_illuminant, spd_to_xyz, xyz_to_lab, delta_e
>>> d65 = get_illuminant("D65")
>>> xyz = spd_to_xyz(d65)
>>> lab = xyz_to_lab(xyz, "D65")
>>> round(lab.L)  # perfect diffuser under its own white → L*≈100
100
"""

from __future__ import annotations

from chromatone.adaptation import (
    AdaptationTransform,
    adapt_xyz,
    adapt_xyz_bradford,
    build_transform,
    chromatic_adaptation_matrix,
)
from chromatone.cmf import CMF, cie_1931_2deg, load_cmf
from chromatone.delta_e import (
    delta_e,
    delta_e_cie76,
    delta_e_cie94,
    delta_e_ciede2000,
    delta_e_xyz,
)
from chromatone.illuminant import (
    blackbody_spd,
    get_illuminant,
    list_illuminants,
)
from chromatone.io import read_spd, write_spd
from chromatone.lab import Lab, LCh, lab_to_xyz, xyz_to_lab
from chromatone.mix import gaussian_led, mix_leds, mix_spds, rgb_led_set
from chromatone.spd import SPD
from chromatone.xyz import XYZ, spd_to_xyz, white_point, xyY

__version__ = "0.1.0"


# Stable fixture catalog (internal golden IDs)
from ._catalog import FIXTURE_NOTES, REFERENCE_URLS, fixture_ids, note_for, primary_reference

__all__ = ["FIXTURE_NOTES", "REFERENCE_URLS", "fixture_ids", "note_for", "primary_reference", 
    "SPD",
    "CMF",
    "XYZ",
    "xyY",
    "Lab",
    "LCh",
    "AdaptationTransform",
    "cie_1931_2deg",
    "load_cmf",
    "get_illuminant",
    "list_illuminants",
    "blackbody_spd",
    "spd_to_xyz",
    "white_point",
    "xyz_to_lab",
    "lab_to_xyz",
    "delta_e",
    "delta_e_cie76",
    "delta_e_cie94",
    "delta_e_ciede2000",
    "delta_e_xyz",
    "adapt_xyz",
    "adapt_xyz_bradford",
    "build_transform",
    "chromatic_adaptation_matrix",
    "read_spd",
    "write_spd",
    "mix_spds",
    "mix_leds",
    "gaussian_led",
    "rgb_led_set",
    "__version__",
]
