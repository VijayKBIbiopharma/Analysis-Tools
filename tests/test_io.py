"""Reading files.

The failures that matter here are quiet ones: a file that reads but reads
*wrongly* produces an analysis of the wrong numbers, and nothing in the output
looks amiss.
"""

import io

import pandas as pd
import pytest

from analysis_tools.io import (
    excel_sheet_names, read_delimited, read_excel, sniff_delimiter,
)


@pytest.mark.parametrize("text,expected", [
    ("a,b,c\n1,2,3\n4,5,6\n", ","),
    ("a\tb\tc\n1\t2\t3\n4\t5\t6\n", "\t"),
    ("a;b;c\n1;2;3\n4;5;6\n", ";"),
    ("a|b|c\n1|2|3\n4|5|6\n", "|"),
    ("only\n1\n2\n3\n", ","),
])
def test_delimiters_are_recognised(text, expected):
    assert sniff_delimiter(text) == expected


def test_a_comma_inside_a_quoted_field_is_not_a_delimiter():
    table = read_delimited(b'Name,Note\n"Smith, J.",fine\n"Jones, A.",also fine\n')
    assert list(table.frame.columns) == ["Name", "Note"]
    assert table.frame.loc[0, "Name"] == "Smith, J."


def test_the_excel_byte_order_mark_does_not_become_part_of_the_first_column_name():
    """Excel writes UTF-8 with a BOM by default. Left in, the first column is
    named '\\ufeffBatch' and every later lookup by name silently misses it."""
    table = read_delimited("﻿Batch,Titer\nB1,2.4\nB2,2.9\n".encode("utf-8"))
    assert list(table.frame.columns) == ["Batch", "Titer"]


def test_windows_western_encoding_is_read_rather_than_rejected():
    table = read_delimited("Site,Yield\nMünchen,2.4\n".encode("cp1252"))
    assert table.frame.loc[0, "Site"] == "München"


def test_unnamed_columns_are_named_for_their_position():
    """pandas calls them 'Unnamed: 2', which reads as data to someone who did
    not write it."""
    table = read_delimited(b"A,,C\n1,2,3\n4,5,6\n")
    assert list(table.frame.columns) == ["A", "Column 2", "C"]
    assert any("Renamed" in n for n in table.notes)


def test_repeated_column_names_are_numbered_and_the_rename_is_reported():
    table = read_delimited(b"Titer,Titer,Titer\n1,2,3\n4,5,6\n")
    assert list(table.frame.columns) == ["Titer", "Titer (2)", "Titer (3)"]
    assert any("Renamed" in n for n in table.notes)


def test_surrounding_whitespace_in_headers_is_stripped():
    table = read_delimited(b"  Batch  ,  Titer  \nB1,2.4\n")
    assert list(table.frame.columns) == ["Batch", "Titer"]


def test_blank_lines_do_not_become_rows():
    table = read_delimited(b"A,B\n1,2\n\n\n3,4\n")
    assert len(table.frame) == 2


def test_the_shape_is_reported_the_way_a_person_reads_it():
    table = read_delimited(b"A,B\n" + b"1,2\n" * 1500)
    assert table.shape_text == "1,500 rows x 2 columns"


# ------------------------------------------------------------------- workbooks

def _workbook(sheets):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def test_a_workbook_lists_its_sheets_without_reading_them():
    data = _workbook({
        "Runs": pd.DataFrame({"A": [1, 2]}),
        "Notes": pd.DataFrame({"B": ["x", "y"]}),
    })
    assert excel_sheet_names(data) == ["Runs", "Notes"]


def test_a_named_sheet_is_the_one_that_gets_read():
    data = _workbook({
        "First": pd.DataFrame({"A": [1, 2]}),
        "Second": pd.DataFrame({"B": [9, 9]}),
    })
    table = read_excel(data, sheet="Second")
    assert list(table.frame.columns) == ["B"]
    assert table.sheet == "Second"


def test_with_no_sheet_named_the_first_is_read():
    data = _workbook({"First": pd.DataFrame({"A": [1]}), "Second": pd.DataFrame({"B": [2]})})
    assert read_excel(data).sheet == "First"


# -------------------------------------------------------------- the disk rule

def test_nothing_in_this_module_opens_a_path():
    """Section A.6 of the ECP asks that applications not rely on local storage,
    and the privacy position is stronger: a file handed to this tool is read
    into memory and never written anywhere. Enforced by taking bytes, never a
    path."""
    import inspect

    from analysis_tools import io as module

    source = inspect.getsource(module)
    assert "open(" not in source
    for entry in (read_delimited, read_excel, excel_sheet_names):
        first = list(inspect.signature(entry).parameters)[0]
        assert first == "data", "%s should take bytes, not a path" % entry.__name__
