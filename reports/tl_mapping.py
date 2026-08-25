"""
reports/tl_mapping.py
Small persisted lookup: Associate/Counsellor name -> TL name.

Uploaded once (CSV or XLSX with an "Associate"/"Counsellor" column and a
"TL"/"Team Lead" column) via /api/v1/tl-mapping/upload, then used by
reports/audit_excel_manager.auto_fill_from_report() to auto-fill the
"TL Name" tracker column whenever "Lead Owner" is known.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TL_MAP_PATH = Path("data/associate_tl_map.json")


def load_tl_map() -> dict:
    if TL_MAP_PATH.exists():
        try:
            return json.loads(TL_MAP_PATH.read_text())
        except Exception as exc:
            logger.warning(f"Could not read TL mapping: {exc}")
    return {}


def save_tl_map(mapping: dict) -> None:
    TL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    TL_MAP_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))


def lookup_tl(associate_name: str) -> str:
    """Case-insensitive lookup of an associate's TL. Returns '' if unknown."""
    if not associate_name:
        return ""
    return load_tl_map().get(associate_name.strip().lower(), "")


def parse_mapping_file(content: bytes, filename: str) -> dict:
    """
    Parse an uploaded CSV/XLSX into {associate_name_lower: tl_name}.
    Auto-detects the associate/TL columns by header name.
    """
    filename = (filename or "").lower()
    if filename.endswith((".xlsx", ".xls")):
        import openpyxl, io
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    else:
        import csv, io
        text = content.decode("utf-8-sig", errors="ignore")
        rows = list(csv.reader(io.StringIO(text)))

    if not rows:
        raise ValueError("File is empty")

    header = [str(h or "").strip().lower() for h in rows[0]]

    def _find(*keywords: str) -> int:
        for i, h in enumerate(header):
            if any(kw in h for kw in keywords):
                return i
        return -1

    assoc_idx = _find("associate", "counsellor", "counselor", "owner", "agent")
    tl_idx    = _find("tl name", "team lead", "tl")
    if assoc_idx == -1:
        assoc_idx = 0
    if tl_idx == -1 or tl_idx == assoc_idx:
        tl_idx = 1 if len(header) > 1 else 0

    mapping: dict = {}
    for row in rows[1:]:
        if not row or assoc_idx >= len(row) or tl_idx >= len(row):
            continue
        assoc = str(row[assoc_idx] or "").strip()
        tl    = str(row[tl_idx] or "").strip()
        if assoc:
            mapping[assoc.lower()] = tl
    return mapping
