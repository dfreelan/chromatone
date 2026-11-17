# chromatone

**Spectral colour science toolkit for imaging, vision, and graphics.**

`chromatone` is a stdlib-only Python 3.10+ library for working with spectral
power distributions (SPDs), CIE colour-matching functions, standard
illuminants, XYZ/Lab conversions, Bradford chromatic adaptation, and
CIE76 / CIE94 / CIEDE2000 colour differences. It is designed for research
and engineering teams that need transparent, dependency-light colourimetry.

## Features

- **SPD** objects with validation, linear interpolation, resampling, and arithmetic
- Embedded **CIE 1931 2°** colour-matching functions (5 nm, 380–780 nm)
- Standard **illuminants**: A, C, D50/D55/D65/D75, E, fluorescent F1–F12
- **Planckian** (blackbody) SPD synthesis
- **CIE XYZ** and **xyY** from spectra (sources or reflectance × illuminant)
- **CIELAB / CIELCh** with XYZ round-trip
- **Bradford**, von Kries / HPE, CAT02, and XYZ-scaling chromatic adaptation
- **ΔE**: CIE76, CIE94, CIEDE2000 (Sharma et al. reference cases in tests)
- Spectrum **CSV / JSON** I/O
- Simple **LED / SPD mixing** and Gaussian channel factories
- CLI entry point: `chromatone`

## Install

```bash
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

No runtime dependencies beyond the Python standard library.

## Quick start

```python
from chromatone import (
    get_illuminant,
    spd_to_xyz,
    xyz_to_lab,
    delta_e,
    Lab,
    adapt_xyz_bradford,
)

# White point of D65
d65 = get_illuminant("D65")
xyz = spd_to_xyz(d65)          # Y ≈ 100
lab = xyz_to_lab(xyz, "D65")   # L* ≈ 100, a*≈0, b*≈0

# Colour difference
dE = delta_e(Lab(50, 2.5, 0), Lab(50, 0, -2.5), method="cie00")

# Adapt illuminant A white toward D65 with Bradford CAT
from chromatone import white_point
adapted = adapt_xyz_bradford(white_point("A"), "A", "D65")
```

### Reflectance under an illuminant

```python
from chromatone import SPD, get_illuminant, spd_to_xyz, xyz_to_lab

# Flat 50% reflectance
refl = SPD.ones().scale(0.5, name="grey50")
xyz = spd_to_xyz(refl, illuminant=get_illuminant("D65"))
lab = xyz_to_lab(xyz, "D65")
```

### Mix LED-like Gaussians

```python
from chromatone import rgb_led_set, mix_leds, spd_to_xyz

leds = rgb_led_set(fwhm_nm=25)
mix = mix_leds(leds, [0.3, 0.4, 0.5])
print(spd_to_xyz(mix))
```

## CLI

```bash
chromatone illuminants
chromatone whitepoint D65
chromatone xyz spectrum.csv
chromatone lab spectrum.json --white D65
chromatone deltae --method cie00 "50,0,0" "50,2,2"
chromatone adapt --from A --to D65 109.85 100 35.58
chromatone blackbody 2856 --out illuminant_a.json
chromatone mix --weights 0.5,0.5 a.csv b.csv --out mix.csv
```

## Package layout

| Module | Role |
|--------|------|
| `spd` | Spectral power distribution type |
| `cmf` | CIE colour-matching functions |
| `illuminant` | Standard illuminants & blackbody |
| `xyz` | Tristimulus integration & chromaticity |
| `lab` | CIELAB / CIELCh |
| `delta_e` | CIE76 / 94 / 00 |
| `adaptation` | Bradford & other CATs |
| `io` | CSV / JSON spectrum I/O |
| `mix` | SPD / LED mixing |
| `cli` | Command-line interface |
| `data/` | Embedded CMF & illuminant tables |

## Colourimetric notes

- CMFs and illuminants are sampled at **5 nm** from **380–780 nm**. SPDs on
  other grids are linearly interpolated onto the CMF support during
  integration (trapezoidal rule).
- White points use **Y = 100** by default (CIE practice for reflectance).
- CIEDE2000 follows the Sharma, Wu & Dalal (2005) notes; the test suite
  includes their published reference pairs.
- Chromatic adaptation defaults to the **Bradford** cone-response matrix.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -q
```

## License

MIT — see [LICENSE](LICENSE).

## References

- CIE 015:2018 *Colorimetry*, 4th edition
- ISO/CIE 11664 series (observers, illuminants, Lab, ΔE)
- Sharma, G., Wu, W., & Dalal, E. N. (2005). The CIEDE2000 color-difference formula
- Wyszecki & Stiles, *Color Science*, 2nd ed. (CMF tables)
