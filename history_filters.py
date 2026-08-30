"""
history_filters.py
Single source of truth for Audit History filter semantics — used by the
history listing endpoint, the dashboard aggregation, and the export
endpoint, so the three can never drift out of sync. Whichever of them
calls build_where() with the same HistoryFilters gets identical matching
rows; there is exactly one filter-to-SQL translation in this codebase.

Field mapping (no schema changes — everything here reads data that
already exists):

  search              -> sessions.label / session_id, plus the CSV-sourced
                         Prospect ID / Lead Number / Lead Name inside
                         lead_sheet_json (ILIKE/substring). Preserves the
                         existing associate/session-id search exactly, and
                         extends it.

  audit_date_from/to  -> sessions.completed_at (when the audit PIPELINE
                         finished — naive UTC text, datetime.utcnow(),
                         same convention already used throughout this
                         codebase). NOT the same thing as the Excel
                         Tracker's own "Audit date" column, which is a
                         display value generated fresh at push-to-tracker
                         time (reports/audit_excel_manager.py,
                         datetime.now(), no persistence per-session) — that
                         field lives only inside the generated workbook,
                         not in the sessions table, and is untouched by
                         this feature.

  demo_date_from/to   -> lead_sheet_json's "Demo Date" — only present for
                         sessions uploaded via POST /api/v1/audit/upload-csv
                         (confirmed empirically: 23/60 real sessions have
                         it, the rest — single-link/pasted-batch uploads —
                         have lead_sheet_json = '{}'). Compared as plain
                         text against the CRM's own stored format
                         ("YYYY-MM-DD HH:MM:SS", space-separated, no
                         timezone marker) — no conversion applied, matching
                         the rest of this app, which has no timezone
                         handling anywhere.

  associate           -> sessions.label. This is ALREADY the app's
                         established canonical associate field:
                         api/main.py's upload_csv_batch sets it from the
                         CSV's "Lead Owner" column specifically (never
                         "Owner"), and reports/associate_analytics.py
                         layers the tracker-push correction on top of it
                         for the same reason ("Session 1 [tracker push] is
                         authoritative once it exists"). "Owner" is a
                         separate field preserved as-is inside
                         lead_sheet_json (real data checked: identical to
                         Lead Owner in every session so far, but not
                         assumed to always be — this filter/search never
                         reads "Owner").

  tl                  -> lead_sheet_json's "TL Name".
  lead_stage          -> lead_sheet_json's "Lead Stage".
  payment_status      -> lead_sheet_json's "Payment Done" ("yes"/"no"), or
                         "unknown" meaning no lead_sheet at all, or the
                         field is present but blank.

No new columns, no migration: every lead_sheet_json field is read via
Postgres JSON operators on the existing TEXT column (lead_sheet_json is
TEXT, not jsonb, on purpose — see session_store.py's _summary_json_exprs,
which this mirrors — so the SQLite and Postgres schemas stay identical),
backed by matching functional indexes (session_store.py's init_db() /
db/postgres_schema.py). SQLite gets the equivalent via json_extract().

Security: _lead_field_expr()'s `key` argument is ALWAYS one of this
module's own fixed literal strings ("TL Name", "Lead Stage", ...) — never
derived from request input. Every actual VALUE a user supplies (search
text, date strings, associate names, ...) is passed as a parameterized
placeholder, never interpolated into the SQL text itself.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

# Whitelisted sort fields only — never accept an arbitrary column name from
# the request (see api/main.py's endpoints, which validate via this dict).
SORTABLE_FIELDS = {
    "audit_date": "completed_at",   # default — matches "Default sort: Audit Date descending"
    "created_at": "created_at",
    "score":      "overall_score",
}
DEFAULT_SORT_BY = "audit_date"
DEFAULT_SORT_ORDER = "desc"

PAYMENT_STATUSES = {"yes", "no", "unknown"}

# The lead_sheet_json keys this module ever extracts — fixed, not
# user-controlled. Kept as named constants (not raw literals scattered
# through build_where) so a rename only happens in one place.
LEAD_FIELD_TL_NAME       = "TL Name"
LEAD_FIELD_LEAD_STAGE    = "Lead Stage"
LEAD_FIELD_PAYMENT_DONE  = "Payment Done"
LEAD_FIELD_DEMO_DATE     = "Demo Date"
LEAD_FIELD_PROSPECT_ID   = "Prospect ID"
LEAD_FIELD_LEAD_NUMBER   = "Lead Number"
LEAD_FIELD_LEAD_NAME     = "Lead Name"


@dataclass
class HistoryFilters:
    search: Optional[str] = None
    audit_date_from: Optional[str] = None   # "YYYY-MM-DD" date-only, from the UI's date picker
    audit_date_to: Optional[str] = None
    demo_date_from: Optional[str] = None
    demo_date_to: Optional[str] = None
    associate: list = field(default_factory=list)
    tl: list = field(default_factory=list)
    lead_stage: list = field(default_factory=list)
    payment_status: Optional[str] = None    # "yes" | "no" | "unknown" | None (= all)
    sort_by: str = DEFAULT_SORT_BY
    sort_order: str = DEFAULT_SORT_ORDER

    def __post_init__(self):
        if self.sort_by not in SORTABLE_FIELDS:
            self.sort_by = DEFAULT_SORT_BY
        if self.sort_order not in ("asc", "desc"):
            self.sort_order = DEFAULT_SORT_ORDER
        if self.payment_status and self.payment_status not in PAYMENT_STATUSES:
            self.payment_status = None

    @property
    def sort_column(self) -> str:
        return SORTABLE_FIELDS[self.sort_by]

    def is_empty(self) -> bool:
        return not any([
            self.search, self.audit_date_from, self.audit_date_to,
            self.demo_date_from, self.demo_date_to, self.associate,
            self.tl, self.lead_stage, self.payment_status,
        ])

    def as_display_lines(self) -> list[tuple[str, str]]:
        """(label, value) pairs for the export Summary sheet — 'All' for
        anything not filtered, matching the export panel's own preview."""
        def _range(frm, to):
            if not frm and not to:
                return "All"
            return f"{frm or '…'} to {to or '…'}"
        return [
            ("Audit Date",     _range(self.audit_date_from, self.audit_date_to)),
            ("Demo Date",      _range(self.demo_date_from, self.demo_date_to)),
            ("Associate(s)",   ", ".join(self.associate) if self.associate else "All"),
            ("TL(s)",          ", ".join(self.tl) if self.tl else "All"),
            ("Lead Stage(s)",  ", ".join(self.lead_stage) if self.lead_stage else "All"),
            ("Payment",        self.payment_status.capitalize() if self.payment_status else "All"),
            ("Search",         self.search or "—"),
        ]


def _as_list(v) -> list:
    """Accepts a list, a comma-separated string, or None/empty — matches
    how repeated query params vs. a single comma-joined param both show up
    depending on how the frontend sends a multi-select."""
    if not v:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return [s.strip() for s in v if s and str(s).strip()]


def parse_history_filters(search=None, audit_date_from=None, audit_date_to=None,
                           demo_date_from=None, demo_date_to=None,
                           associate=None, tl=None, lead_stage=None,
                           payment_status=None, sort_by=None, sort_order=None) -> HistoryFilters:
    """Normalizes raw (query-param-shaped) input into a HistoryFilters."""
    return HistoryFilters(
        search=(search or "").strip() or None,
        audit_date_from=(audit_date_from or "").strip() or None,
        audit_date_to=(audit_date_to or "").strip() or None,
        demo_date_from=(demo_date_from or "").strip() or None,
        demo_date_to=(demo_date_to or "").strip() or None,
        associate=_as_list(associate),
        tl=_as_list(tl),
        lead_stage=_as_list(lead_stage),
        payment_status=(payment_status or "").strip().lower() or None,
        sort_by=(sort_by or DEFAULT_SORT_BY).strip(),
        sort_order=(sort_order or DEFAULT_SORT_ORDER).strip().lower(),
    )


def _lead_field_expr(json_col: str, key: str) -> str:
    """Backend-aware SQL expression extracting one lead_sheet_json field as
    plain text. `key` must always be one of this module's own fixed
    constants (never user input) — see module docstring's Security note."""
    from db.backend import is_postgres_enabled
    if is_postgres_enabled():
        escaped = key.replace("'", "''")
        return f"({json_col}::jsonb ->> '{escaped}')"
    escaped = key.replace('"', '\\"')
    return f"json_extract({json_col}, '$.\"{escaped}\"')"


def next_day(date_str: str) -> str:
    """'2026-08-30' -> '2026-08-31' — an inclusive-end date range filter is
    built as >= start AND < day-after-end, so a record from later on the
    end day is never accidentally excluded (see module docstring)."""
    y, m, d = (int(x) for x in date_str.split("-"))
    return (date(y, m, d) + timedelta(days=1)).isoformat()


def build_where(filters: HistoryFilters) -> tuple[str, list]:
    """Returns (conditions_joined_by_AND, params) — empty string/list if no
    filter is active. Caller prefixes with 'WHERE ' only when non-empty."""
    if filters is None:
        return "", []

    conditions: list[str] = []
    params: list = []

    if filters.search:
        like = f"%{filters.search.lower()}%"
        prospect_expr = _lead_field_expr("lead_sheet_json", LEAD_FIELD_PROSPECT_ID)
        lead_number_expr = _lead_field_expr("lead_sheet_json", LEAD_FIELD_LEAD_NUMBER)
        lead_name_expr = _lead_field_expr("lead_sheet_json", LEAD_FIELD_LEAD_NAME)
        conditions.append(
            "(LOWER(label) LIKE ? OR LOWER(session_id) LIKE ? "
            f"OR LOWER(COALESCE({prospect_expr}, '')) LIKE ? "
            f"OR LOWER(COALESCE({lead_number_expr}, '')) LIKE ? "
            f"OR LOWER(COALESCE({lead_name_expr}, '')) LIKE ?)"
        )
        params += [like, like, like, like, like]

    if filters.audit_date_from:
        conditions.append("completed_at >= ?")
        params.append(f"{filters.audit_date_from}T00:00:00")
    if filters.audit_date_to:
        conditions.append("completed_at < ?")
        params.append(f"{next_day(filters.audit_date_to)}T00:00:00")

    if filters.demo_date_from or filters.demo_date_to:
        demo_expr = _lead_field_expr("lead_sheet_json", LEAD_FIELD_DEMO_DATE)
        if filters.demo_date_from:
            conditions.append(f"{demo_expr} >= ?")
            params.append(f"{filters.demo_date_from} 00:00:00")
        if filters.demo_date_to:
            conditions.append(f"{demo_expr} < ?")
            params.append(f"{next_day(filters.demo_date_to)} 00:00:00")

    if filters.associate:
        placeholders = ",".join(["?"] * len(filters.associate))
        conditions.append(f"label IN ({placeholders})")
        params += filters.associate

    if filters.tl:
        tl_expr = _lead_field_expr("lead_sheet_json", LEAD_FIELD_TL_NAME)
        placeholders = ",".join(["?"] * len(filters.tl))
        conditions.append(f"{tl_expr} IN ({placeholders})")
        params += filters.tl

    if filters.lead_stage:
        stage_expr = _lead_field_expr("lead_sheet_json", LEAD_FIELD_LEAD_STAGE)
        placeholders = ",".join(["?"] * len(filters.lead_stage))
        conditions.append(f"{stage_expr} IN ({placeholders})")
        params += filters.lead_stage

    if filters.payment_status:
        pay_expr = _lead_field_expr("lead_sheet_json", LEAD_FIELD_PAYMENT_DONE)
        if filters.payment_status == "unknown":
            conditions.append(
                f"(lead_sheet_json IS NULL OR lead_sheet_json = '{{}}' "
                f"OR {pay_expr} IS NULL OR {pay_expr} = '')"
            )
        else:
            conditions.append(f"LOWER({pay_expr}) = ?")
            params.append(filters.payment_status)

    return " AND ".join(conditions), params
