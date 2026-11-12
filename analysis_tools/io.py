"""Reading a table out of whatever the user has.

Scope: delimited text (CSV, TSV, semicolon, pipe) and Excel workbooks. Those are
what process data actually arrives as. Nothing here writes to disk.

The ECP asks (section A.6) that applications not rely on local storage, and the
privacy position this tool takes is stronger than that: a file handed to it is
read into memory and never written anywhere. Both are served by the same rule,
so it is enforced here rather than left to each caller - every entry point takes
bytes or a file-like object, never a path it opens for itself.
"""

from __future__ import annotations

import csv
import io as _io
from dataclasses import dataclass
from typing import Optional

import pandas as pd

DELIMITERS = [",", "\t", ";", "|"]
MAX_PREVIEW_BYTES = 64 * 1024


@dataclass
class LoadedTable:
    """A table, and an account of how it was read."""

    frame: pd.DataFrame
    source_name: str
    sheet: Optional[str] = None
    delimiter: Optional[str] = None
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    @property
    def shape_text(self) -> str:
        rows, cols = self.frame.shape
        return "%s rows x %s columns" % (f"{rows:,}", cols)


def sniff_delimiter(sample: str) -> str:
    """Which delimiter this text uses.

    csv.Sniffer first, because it handles quoted fields properly. It raises on
    text it cannot read, and a single-column file is exactly that - so fall back
    to counting, which gives the comma default a single-column file needs.
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        pass
    first = next((line for line in sample.splitlines() if line.strip()), "")
    best, best_count = ",", 0
    for delimiter in DELIMITERS:
        count, in_quotes = 0, False
        for ch in first:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == delimiter and not in_quotes:
                count += 1
        if count > best_count:
            best, best_count = delimiter, count
    return best


def read_delimited(data: bytes, source_name: str = "data") -> LoadedTable:
    """Read delimited text from bytes held in memory."""
    text = _decode(data)
    delimiter = sniff_delimiter(text[:MAX_PREVIEW_BYTES])
    frame = pd.read_csv(
        _io.StringIO(text), sep=delimiter, skip_blank_lines=True,
        skipinitialspace=True,
    )
    notes = []
    frame, renamed = _tidy_columns(frame, raw_header=_raw_header(text, delimiter))
    if renamed:
        notes.append("Renamed %d unnamed or duplicated column(s): %s"
                     % (len(renamed), ", ".join(renamed[:6])))
    return LoadedTable(frame=frame, source_name=source_name,
                       delimiter=delimiter, notes=notes)


def excel_sheet_names(data: bytes) -> list:
    """The sheets in a workbook, without reading any of them."""
    with pd.ExcelFile(_io.BytesIO(data)) as book:
        return list(book.sheet_names)


def read_excel(data: bytes, sheet: Optional[str] = None,
               source_name: str = "workbook") -> LoadedTable:
    """Read one sheet of a workbook held in memory.

    Macros are never executed: pandas reads the cell values through openpyxl,
    which does not run VBA.
    """
    with pd.ExcelFile(_io.BytesIO(data)) as book:
        chosen = sheet if sheet is not None else book.sheet_names[0]
        frame = book.parse(chosen)
        header = book.parse(chosen, header=None, nrows=1)
    raw = [None if pd.isna(v) else str(v) for v in header.iloc[0].tolist()] if len(header) else None
    notes = []
    frame, renamed = _tidy_columns(frame, raw_header=raw)
    if renamed:
        notes.append("Renamed %d unnamed or duplicated column(s): %s"
                     % (len(renamed), ", ".join(renamed[:6])))
    return LoadedTable(frame=frame, source_name=source_name, sheet=chosen, notes=notes)


def _decode(data: bytes) -> str:
    """Bytes to text, tolerating what spreadsheets emit.

    Excel writes UTF-8 with a byte-order mark by default and cp1252 when told to
    save as CSV on a Windows machine with a Western locale. Both turn up.
    """
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _raw_header(text: str, delimiter: str):
    """The header row exactly as the file wrote it.

    pandas resolves duplicate headers before handing them over - three columns
    called Titer arrive as Titer, Titer.1, Titer.2 - so by the time the frame
    exists the duplication is invisible and ".1" looks like part of the name.
    Reading the header row directly is the only way to know it happened, and the
    user needs to know: two columns with the same name usually means a join went
    wrong upstream.
    """
    for line in text.splitlines():
        if line.strip():
            return next(csv.reader([line], delimiter=delimiter), None)
    return None


def _tidy_columns(frame: pd.DataFrame, raw_header=None):
    """Name the unnamed, number the duplicated, strip the rest.

    pandas calls an unnamed column "Unnamed: 3" and appends ".1" to a repeat.
    Both read as data to a user who did not write them, so both are replaced
    with something that says what happened. When the file's own header row is
    available and has the same number of fields, it is used in preference to
    what pandas made of it.
    """
    source = list(frame.columns)
    if raw_header is not None and len(raw_header) == len(source):
        source = raw_header
    renamed, seen, out = [], {}, []
    for i, name in enumerate(source):
        text = "" if name is None else str(name).strip()
        if text == "" or text.lower().startswith("unnamed:"):
            text = "Column %d" % (i + 1)
            renamed.append(text)
        base = text
        if base in seen:
            seen[base] += 1
            text = "%s (%d)" % (base, seen[base])
            renamed.append(text)
        else:
            seen[base] = 1
        out.append(text)
    frame = frame.copy()
    frame.columns = out
    return frame, renamed
