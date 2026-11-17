"""Command-line interface for chromatone."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from chromatone import __version__
from chromatone.adaptation import adapt_xyz, available_methods
from chromatone.delta_e import delta_e
from chromatone.illuminant import (
    blackbody_spd,
    get_illuminant,
    illuminant_description,
    list_illuminants,
)
from chromatone.io import read_spd, write_spd
from chromatone.lab import Lab, lab_to_xyz, xyz_to_lab
from chromatone.mix import mix_spds
from chromatone.spd import SPD
from chromatone.xyz import XYZ, spd_to_xyz, white_point


def _parse_xyz(text: str) -> XYZ:
    parts = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit(f"expected 3 XYZ components, got {text!r}")
    return XYZ(float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_lab(text: str) -> Lab:
    parts = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit(f"expected 3 Lab components, got {text!r}")
    return Lab(float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_weights(text: Optional[str], n: int) -> list[float]:
    if text is None:
        return [1.0] * n
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != n:
        raise SystemExit(f"expected {n} weights, got {len(parts)}")
    return [float(p) for p in parts]


def cmd_illuminants(_: argparse.Namespace) -> int:
    for name in list_illuminants():
        print(f"{name:6s}  {illuminant_description(name)}")
    return 0


def cmd_whitepoint(args: argparse.Namespace) -> int:
    xyz = white_point(args.illuminant, normalize_y=args.y)
    xy = xyz.to_xy()
    print(f"Illuminant: {args.illuminant}")
    print(f"XYZ: {xyz.X:.6f} {xyz.Y:.6f} {xyz.Z:.6f}")
    print(f"xy:  {xy[0]:.6f} {xy[1]:.6f}")
    return 0


def cmd_xyz(args: argparse.Namespace) -> int:
    spd = read_spd(args.spectrum)
    illuminant = get_illuminant(args.illuminant) if args.as_reflectance else None
    xyz = spd_to_xyz(
        spd,
        illuminant=illuminant,
        normalize_y=None if args.raw else args.y,
    )
    print(f"SPD: {spd}")
    print(f"XYZ: {xyz.X:.6f} {xyz.Y:.6f} {xyz.Z:.6f}")
    try:
        xy = xyz.to_xy()
        print(f"xy:  {xy[0]:.6f} {xy[1]:.6f}")
    except Exception:
        pass
    return 0


def cmd_lab(args: argparse.Namespace) -> int:
    spd = read_spd(args.spectrum)
    illuminant_spd = get_illuminant(args.illuminant) if args.as_reflectance else None
    xyz = spd_to_xyz(
        spd,
        illuminant=illuminant_spd,
        normalize_y=None if args.raw else 100.0,
    )
    lab = xyz_to_lab(xyz, args.white)
    lch = lab.to_lch()
    print(f"SPD: {spd}")
    print(f"XYZ: {xyz.X:.6f} {xyz.Y:.6f} {xyz.Z:.6f}")
    print(f"Lab: {lab.L:.6f} {lab.a:.6f} {lab.b:.6f}")
    print(f"LCh: {lch.L:.6f} {lch.C:.6f} {lch.h:.6f}")
    return 0


def cmd_deltae(args: argparse.Namespace) -> int:
    lab1 = _parse_lab(args.lab1)
    lab2 = _parse_lab(args.lab2)
    d = delta_e(lab1, lab2, method=args.method)
    print(f"Lab1: {lab1}")
    print(f"Lab2: {lab2}")
    print(f"ΔE ({args.method}): {d:.6f}")
    return 0


def cmd_adapt(args: argparse.Namespace) -> int:
    xyz = _parse_xyz(args.xyz)
    out = adapt_xyz(xyz, args.source, args.destination, method=args.method)
    print(f"Source white: {args.source}")
    print(f"Dest white:   {args.destination}")
    print(f"Method:       {args.method}")
    print(f"In  XYZ: {xyz.X:.6f} {xyz.Y:.6f} {xyz.Z:.6f}")
    print(f"Out XYZ: {out.X:.6f} {out.Y:.6f} {out.Z:.6f}")
    return 0


def cmd_mix(args: argparse.Namespace) -> int:
    spectra = [read_spd(p) for p in args.spectra]
    weights = _parse_weights(args.weights, len(spectra))
    mixed = mix_spds(spectra, weights, name=args.name, normalize_weights=args.normalize)
    if args.out:
        write_spd(mixed, args.out)
        print(f"Wrote mixed SPD to {args.out}")
    else:
        print(mixed)
        xyz = spd_to_xyz(mixed)
        print(f"XYZ: {xyz.X:.6f} {xyz.Y:.6f} {xyz.Z:.6f}")
    return 0


def cmd_blackbody(args: argparse.Namespace) -> int:
    spd = blackbody_spd(
        args.temperature,
        start_nm=args.start,
        stop_nm=args.stop,
        step_nm=args.step,
        normalize_560=not args.no_normalize,
    )
    if args.out:
        write_spd(spd, args.out)
        print(f"Wrote blackbody SPD ({args.temperature} K) to {args.out}")
    else:
        xyz = spd_to_xyz(spd)
        print(spd)
        print(f"XYZ: {xyz.X:.6f} {xyz.Y:.6f} {xyz.Z:.6f}")
        print(f"xy:  {xyz.to_xy()[0]:.6f} {xyz.to_xy()[1]:.6f}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    spd = read_spd(args.spectrum)
    print(f"name:        {spd.name}")
    print(f"samples:     {len(spd)}")
    print(f"range_nm:    {spd.start_nm:.4g} – {spd.stop_nm:.4g}")
    print(f"regular:     {spd.is_regular}")
    if spd.step_nm is not None:
        print(f"step_nm:     {spd.step_nm:.4g}")
    print(f"peak_value:  {max(spd.values):.6g}")
    print(f"integral:    {spd.integral():.6g}")
    return 0


def cmd_export_illuminant(args: argparse.Namespace) -> int:
    spd = get_illuminant(args.name)
    write_spd(spd, args.out)
    print(f"Wrote illuminant {spd.name} to {args.out}")
    return 0


def cmd_convert_lab_xyz(args: argparse.Namespace) -> int:
    if args.direction == "lab2xyz":
        lab = _parse_lab(args.values)
        xyz = lab_to_xyz(lab, args.white)
        print(f"XYZ: {xyz.X:.6f} {xyz.Y:.6f} {xyz.Z:.6f}")
    else:
        xyz = _parse_xyz(args.values)
        lab = xyz_to_lab(xyz, args.white)
        print(f"Lab: {lab.L:.6f} {lab.a:.6f} {lab.b:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chromatone",
        description="Spectral colour science toolkit (SPD, XYZ, Lab, delta-E).",
    )
    parser.add_argument("--version", action="version", version=f"chromatone {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("illuminants", help="List embedded CIE illuminants")
    p.set_defaults(func=cmd_illuminants)

    p = sub.add_parser("whitepoint", help="Print XYZ/xy of an illuminant white point")
    p.add_argument("illuminant", nargs="?", default="D65")
    p.add_argument("--y", type=float, default=100.0, help="Normalize Y to this value")
    p.set_defaults(func=cmd_whitepoint)

    p = sub.add_parser("xyz", help="Compute XYZ for a spectrum file")
    p.add_argument("spectrum", help="CSV/JSON spectrum path")
    p.add_argument("--illuminant", default="D65", help="Illuminant when --as-reflectance")
    p.add_argument("--as-reflectance", action="store_true")
    p.add_argument("--raw", action="store_true", help="Skip Y normalization")
    p.add_argument("--y", type=float, default=100.0)
    p.set_defaults(func=cmd_xyz)

    p = sub.add_parser("lab", help="Compute Lab/LCh for a spectrum file")
    p.add_argument("spectrum")
    p.add_argument("--illuminant", default="D65")
    p.add_argument("--white", default="D65", help="Lab reference white")
    p.add_argument("--as-reflectance", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.set_defaults(func=cmd_lab)

    p = sub.add_parser("deltae", help="Colour difference between two Lab triples")
    p.add_argument("lab1", help='e.g. "50,0,0"')
    p.add_argument("lab2", help='e.g. "50,2,-1"')
    p.add_argument("--method", default="cie00", choices=["cie76", "cie94", "cie00"])
    p.set_defaults(func=cmd_deltae)

    p = sub.add_parser("adapt", help="Chromatic adaptation of an XYZ triple")
    p.add_argument("xyz", help='e.g. "95,100,108"')
    p.add_argument("--from", dest="source", default="A", help="Source illuminant / white")
    p.add_argument("--to", dest="destination", default="D65", help="Destination white")
    p.add_argument(
        "--method",
        default="bradford",
        choices=available_methods(),
    )
    p.set_defaults(func=cmd_adapt)

    p = sub.add_parser("mix", help="Linearly mix spectrum files")
    p.add_argument("spectra", nargs="+")
    p.add_argument("--weights", help="Comma-separated weights")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--name", default="mix")
    p.add_argument("--out", help="Output CSV/JSON path")
    p.set_defaults(func=cmd_mix)

    p = sub.add_parser("blackbody", help="Synthesize a Planckian SPD")
    p.add_argument("temperature", type=float, help="CCT in kelvin")
    p.add_argument("--start", type=float, default=380.0)
    p.add_argument("--stop", type=float, default=780.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_blackbody)

    p = sub.add_parser("info", help="Summarize a spectrum file")
    p.add_argument("spectrum")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("export-illuminant", help="Write a standard illuminant to a file")
    p.add_argument("name")
    p.add_argument("out")
    p.set_defaults(func=cmd_export_illuminant)

    p = sub.add_parser("convert", help="Convert between Lab and XYZ triples")
    p.add_argument("direction", choices=["lab2xyz", "xyz2lab"])
    p.add_argument("values", help='Comma-separated components, e.g. "50,0,0"')
    p.add_argument("--white", default="D65")
    p.set_defaults(func=cmd_convert_lab_xyz)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 0
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
