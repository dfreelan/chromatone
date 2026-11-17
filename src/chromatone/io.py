"""Spectrum I/O helpers (CSV and JSON)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO, Union

from chromatone.spd import SPD, SPDError

PathLike = Union[str, Path]


class IOError_(ValueError):
    """Raised for spectrum file format errors.

    Named ``IOError_`` to avoid shadowing the builtin ``OSError`` alias.
    """


def _open_text(path: PathLike, mode: str = "r") -> TextIO:
    return open(path, mode, encoding="utf-8", newline="")


def read_spd_csv(
    source: Union[PathLike, TextIO, str],
    *,
    name: Optional[str] = None,
    wavelength_column: int = 0,
    value_column: int = 1,
    delimiter: Optional[str] = None,
) -> SPD:
    """Read an SPD from CSV text or a filesystem path."""
    owns_handle = False
    label = name
    if isinstance(source, (str, Path)) and not (
        isinstance(source, str) and ("\n" in source or "\r" in source)
    ):
        path = Path(source)
        handle: TextIO = _open_text(path, "r")
        owns_handle = True
        if label is None:
            label = path.stem
        text_stream = handle
    elif isinstance(source, str):
        text_stream = io.StringIO(source)
    else:
        text_stream = source  # type: ignore[assignment]

    try:
        sample = text_stream.read(4096)
        text_stream.seek(0)
        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
                delim = dialect.delimiter
            except csv.Error:
                delim = ","
        else:
            delim = delimiter

        reader = csv.reader(text_stream, delimiter=delim)
        pairs: list[tuple[float, float]] = []
        for row_num, row in enumerate(reader, start=1):
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            # Skip header-like rows.
            try:
                float(row[wavelength_column].strip())
            except (ValueError, IndexError):
                continue
            try:
                w = float(row[wavelength_column].strip())
                v = float(row[value_column].strip())
            except (ValueError, IndexError) as exc:
                raise IOError_(f"CSV row {row_num}: {exc}") from exc
            pairs.append((w, v))
        if len(pairs) < 2:
            raise IOError_("CSV must contain at least two numeric samples")
        return SPD.from_pairs(pairs, name=label)
    finally:
        if owns_handle:
            text_stream.close()


def write_spd_csv(
    spd: SPD,
    destination: Union[PathLike, TextIO],
    *,
    header: bool = True,
    delimiter: str = ",",
    precision: int = 6,
) -> None:
    """Write an SPD to CSV."""
    owns_handle = False
    if isinstance(destination, (str, Path)):
        handle: TextIO = _open_text(destination, "w")
        owns_handle = True
        stream = handle
    else:
        stream = destination

    try:
        writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
        if header:
            writer.writerow(["wavelength_nm", "value"])
        fmt = f"{{:.{precision}g}}"
        for w, v in spd:
            writer.writerow([fmt.format(w), fmt.format(v)])
    finally:
        if owns_handle:
            stream.close()


def read_spd_json(
    source: Union[PathLike, TextIO, str, Mapping[str, Any]],
    *,
    name: Optional[str] = None,
) -> SPD:
    """Read an SPD from JSON text, path, or already-parsed mapping."""
    label = name
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        owns_handle = False
        if isinstance(source, (str, Path)) and not (
            isinstance(source, str) and source.lstrip().startswith("{")
        ):
            path = Path(source)
            handle = _open_text(path, "r")
            owns_handle = True
            if label is None:
                label = path.stem
            try:
                data = json.load(handle)
            finally:
                if owns_handle:
                    handle.close()
        elif isinstance(source, str):
            data = json.loads(source)
        else:
            data = json.load(source)  # type: ignore[arg-type]

    if not isinstance(data, dict):
        raise IOError_("JSON spectrum root must be an object")
    if label is not None and "name" not in data:
        data = {**data, "name": label}
    try:
        return SPD.from_dict(data)
    except SPDError as exc:
        raise IOError_(str(exc)) from exc


def write_spd_json(
    spd: SPD,
    destination: Union[PathLike, TextIO],
    *,
    indent: int = 2,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write an SPD to JSON."""
    payload = spd.to_dict()
    if metadata:
        payload["metadata"] = dict(metadata)

    owns_handle = False
    if isinstance(destination, (str, Path)):
        handle = _open_text(destination, "w")
        owns_handle = True
        stream: TextIO = handle
    else:
        stream = destination

    try:
        json.dump(payload, stream, indent=indent)
        stream.write("\n")
    finally:
        if owns_handle:
            stream.close()


def read_spd(
    path: PathLike,
    *,
    format: Optional[str] = None,
    name: Optional[str] = None,
) -> SPD:
    """Read an SPD, inferring format from the file extension when needed.

    Supported extensions: ``.csv``, ``.txt``, ``.json``.
    """
    p = Path(path)
    fmt = (format or p.suffix.lstrip(".")).lower()
    if fmt in {"csv", "txt", "tsv"}:
        return read_spd_csv(p, name=name)
    if fmt in {"json"}:
        return read_spd_json(p, name=name)
    raise IOError_(f"unsupported spectrum format {fmt!r} for {p}")


def write_spd(
    spd: SPD,
    path: PathLike,
    *,
    format: Optional[str] = None,
) -> None:
    """Write an SPD, inferring format from the file extension when needed."""
    p = Path(path)
    fmt = (format or p.suffix.lstrip(".")).lower()
    if fmt in {"csv", "txt", "tsv"}:
        write_spd_csv(spd, p)
        return
    if fmt in {"json"}:
        write_spd_json(spd, p)
        return
    raise IOError_(f"unsupported spectrum format {fmt!r} for {p}")


def spd_to_csv_string(spd: SPD, *, header: bool = True, precision: int = 6) -> str:
    """Serialize an SPD to a CSV string."""
    buf = io.StringIO()
    write_spd_csv(spd, buf, header=header, precision=precision)
    return buf.getvalue()


def spd_to_json_string(spd: SPD, *, indent: int = 2) -> str:
    """Serialize an SPD to a JSON string."""
    return json.dumps(spd.to_dict(), indent=indent) + "\n"
