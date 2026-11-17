"""I/O, mixing, and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chromatone.cli import main
from chromatone.io import (
    read_spd,
    read_spd_csv,
    read_spd_json,
    spd_to_csv_string,
    spd_to_json_string,
    write_spd,
)
from chromatone.mix import (
    MixError,
    convex_combine,
    gaussian_led,
    match_xyz_with_leds,
    mix_leds,
    mix_spds,
    rgb_led_set,
    scale_to_luminance,
)
from chromatone.spd import SPD
from chromatone.xyz import spd_to_xyz, white_point


def test_csv_roundtrip(tmp_path: Path, d65: SPD):
    path = tmp_path / "d65.csv"
    write_spd(d65, path)
    loaded = read_spd(path)
    assert loaded.wavelengths == d65.wavelengths
    assert loaded.values == pytest.approx(d65.values)


def test_json_roundtrip(tmp_path: Path):
    spd = SPD.gaussian(520, 25, name="green")
    path = tmp_path / "green.json"
    write_spd(spd, path)
    loaded = read_spd_json(path)
    assert loaded.name == "green"
    assert loaded.sample(520) == pytest.approx(1.0)


def test_csv_string_and_comments():
    text = "# comment\nwavelength_nm,value\n400,0.1\n500,0.5\n600,0.2\n"
    spd = read_spd_csv(text, name="s")
    assert len(spd) == 3
    assert spd.name == "s"
    assert "wavelength_nm" in spd_to_csv_string(spd)


def test_json_string():
    spd = SPD.ones(name="E")
    data = json.loads(spd_to_json_string(spd))
    assert data["name"] == "E"
    assert len(data["values"]) == len(spd)


def test_mix_spds_weights(d65, illuminant_a):
    mixed = mix_spds([d65, illuminant_a], [0.5, 0.5], normalize_weights=True)
    assert mixed.integral() == pytest.approx(
        0.5 * d65.integral() + 0.5 * illuminant_a.integral(), rel=1e-6
    )


def test_mix_rejects_bad_weights(d65):
    with pytest.raises(MixError):
        mix_spds([d65], [1.0, 2.0])
    with pytest.raises(MixError):
        mix_spds([d65], [-1.0])


def test_led_set_and_mix(led_channels):
    assert len(led_channels) == 3
    mixed = mix_leds(led_channels, [1.0, 1.0, 1.0])
    xyz = spd_to_xyz(mixed)
    assert xyz.Y > 0


def test_gaussian_led_channel():
    ch = gaussian_led(630, 18, name="R")
    assert ch.peak_nm == 630
    assert ch.spd.sample(630) == pytest.approx(1.0)


def test_convex_combine():
    a = SPD.ones()
    b = SPD.zeros()
    mid = convex_combine(a, b, 0.25)
    assert mid.values[0] == pytest.approx(0.75)


def test_scale_to_luminance(d65):
    scaled = scale_to_luminance(d65, 50.0)
    assert spd_to_xyz(scaled, normalize_y=None).Y == pytest.approx(50.0)


def test_match_xyz_with_leds_improves(led_channels):
    # Match a moderate in-gamut XYZ using the three LED channels.
    target = spd_to_xyz(mix_leds(led_channels, [0.4, 0.8, 0.3]), normalize_y=None)
    mixed, weights, achieved = match_xyz_with_leds(
        target, led_channels, iterations=120, learning_rate=0.05
    )
    assert all(w >= -1e-9 for w in weights)
    assert sum(weights) > 0
    assert achieved.Y == pytest.approx(target.Y, rel=0.25)


def test_cli_illuminants(capsys):
    assert main(["illuminants"]) == 0
    out = capsys.readouterr().out
    assert "D65" in out and "FL2" in out


def test_cli_whitepoint(capsys):
    assert main(["whitepoint", "D65"]) == 0
    out = capsys.readouterr().out
    assert "XYZ:" in out


def test_cli_deltae(capsys):
    assert main(["deltae", "50,0,0", "50,3,4", "--method", "cie76"]) == 0
    out = capsys.readouterr().out
    assert "5.000000" in out


def test_cli_adapt(capsys):
    wp = white_point("A")
    assert (
        main(
            [
                "adapt",
                f"{wp.X},{wp.Y},{wp.Z}",
                "--from",
                "A",
                "--to",
                "D65",
            ]
        )
        == 0
    )


def test_cli_blackbody_and_info(tmp_path: Path, capsys):
    out = tmp_path / "bb.json"
    assert main(["blackbody", "5000", "--out", str(out)]) == 0
    assert out.exists()
    assert main(["info", str(out)]) == 0
    assert "samples" in capsys.readouterr().out


def test_cli_xyz_lab(tmp_path: Path, d65, capsys):
    path = tmp_path / "d65.csv"
    write_spd(d65, path)
    assert main(["xyz", str(path)]) == 0
    assert main(["lab", str(path)]) == 0


def test_cli_export_and_mix(tmp_path: Path):
    a = tmp_path / "A.csv"
    b = tmp_path / "D65.csv"
    mixed = tmp_path / "mix.json"
    assert main(["export-illuminant", "A", str(a)]) == 0
    assert main(["export-illuminant", "D65", str(b)]) == 0
    assert main(["mix", str(a), str(b), "--weights", "0.3,0.7", "--out", str(mixed)]) == 0
    assert read_spd(mixed).integral() > 0


def test_cli_convert(capsys):
    assert main(["convert", "xyz2lab", "95.04,100,108.88", "--white", "D65"]) == 0
    assert "Lab:" in capsys.readouterr().out
