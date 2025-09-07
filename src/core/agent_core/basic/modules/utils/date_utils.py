from dataclasses import dataclass
from datetime import date as GDate, timedelta
from typing import Optional, Tuple, Dict, Any, List, Iterable

import jdatetime as jd

'''
Function: convert_calendar(source_calendar, target_calendar, day, month, year)

Input format:

 All args are strings. Use `""` for missing parts.
 source_calendar / target_calendar: "solar" or "gregorian" (normalized).
 day: "01"–"31" or "".
 month: "01"–"12" or "".
 year: "YYYY" (4 digits) or "".
 Only these templates are valid:

  1. DD-MM-YYYY (full date)
  2. MM-YYYY
  3. DD-MM
  4. DD-YYYY
  5. DD
  6. MM
  7. YYYY

Output format:

JSON dict with:

  kind: "exact", "range", "range-approx", "approx", or "invalid".
  source_calendar, target_calendar: normalized names.
  parsed: "dd-mm-yyyy" with blanks for missing parts.
  date:

    * "exact": string "dd-mm-yyyy" or list of 12 strings/"".
    * "range" / "range-approx"`: {"from": "...", "to": "..."}.
    * "approx": {"options": [...]} or list of 12 strings/"".
    * "invalid": {"reason": "..."}.
    * standard_format: always "dd-mm-yyyy".
'''

# Language & digit normalization

PERSIAN_DIGITS = dict(zip("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
ARABIC_INDIC_DIGITS = dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789"))

EN_GREG_MONTHS = {
    "january": 1,   "jan": 1,
    "february": 2,  "feb": 2,
    "march": 3,     "mar": 3,
    "april": 4,     "apr": 4,
    "may": 5,
    "june": 6,      "jun": 6,
    "july": 7,      "jul": 7,
    "august": 8,    "aug": 8,
    "september": 9, "sep": 9,   "sept": 9,
    "october": 10,  "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
FA_GREG_MONTHS = {
    "ژانویه": 1, "فوریه": 2, "مارس": 3, "آوریل": 4, "مه": 5, "می": 5,
    "ژوئن": 6, "ژوئیه": 7, "جولای": 7, "اوت": 8, "آگوست": 8,
    "سپتامبر": 9, "اکتبر": 10, "نوامبر": 11, "دسامبر": 12,
}
FA_SOLAR_MONTHS = {
    "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4, "مرداد": 5, "شهریور": 6,
    "مهر": 7, "آبان": 8, "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12,
}
EN_SOLAR_MONTHS = {
    "farvardin": 1, "ordibehesht": 2, "khordad": 3, "tir": 4, "mordad": 5, "shahrivar": 6,
    "mehr": 7, "aban": 8, "azar": 9, "dey": 10, "day": 10, "bahman": 11, "esfand": 12,
}

_FA_SOLAR = {"شمسی", "خورشیدی", "جلالی", "هجری شمسی"}
_FA_GREG  = {"میلادی", "گریگوری", "گرگوری", "گرگوریان"}

def _normalize_digits(s: str) -> str:
    if not s:
        return s
    out = []
    for ch in s:
        if ch in PERSIAN_DIGITS:
            out.append(PERSIAN_DIGITS[ch])
        elif ch in ARABIC_INDIC_DIGITS:
            out.append(ARABIC_INDIC_DIGITS[ch])
        else:
            out.append(ch)
    return "".join(out)

def _ascii_lower_if_latin(s: str) -> str:
    return s.lower() if any("a" <= c <= "z" or "A" <= c <= "Z" for c in s) else s

def _norm_cal_name(s: str) -> Optional[str]:
    if not s:
        return None
    k = _ascii_lower_if_latin(_normalize_digits(s.strip()))
    if k in {"solar", "jalali", "shamsi", "persian"} or s in _FA_SOLAR:
        return "solar"
    if k in {"gregorian", "greg", "miladi"} or s in _FA_GREG:
        return "gregorian"
    return None

def _parse_int(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    t = _normalize_digits(str(token)).strip()
    if not t or not t.isdigit():
        return None
    v = int(t)
    return v if v > 0 else None

def _parse_month_token(token: Optional[str], calendar_kind: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    t_raw = str(token).strip()
    if not t_raw:
        return None
    t = _normalize_digits(t_raw)
    if t.isdigit():
        v = int(t)
        return v if 1 <= v <= 12 else None
    t_ascii = _ascii_lower_if_latin(t)
    sets: List[Dict[str,int]]
    if calendar_kind == "gregorian":
        sets = [EN_GREG_MONTHS, FA_GREG_MONTHS]
    elif calendar_kind == "solar":
        sets = [EN_SOLAR_MONTHS, FA_SOLAR_MONTHS]
    else:
        sets = [EN_GREG_MONTHS, FA_GREG_MONTHS, EN_SOLAR_MONTHS, FA_SOLAR_MONTHS]
    for m in sets:
        if t_ascii in m: return m[t_ascii]
        if t in m: return m[t]
    return None

def _fmt_dd_mm_yyyy(d: int, m: int, y: int) -> str:
    return f"{d:02d}-{m:02d}-{y:04d}"

def _fmt_dd_mm(d: int, m: int) -> str:
    return f"{d:02d}-{m:02d}"

def _parsed_as_str(d: Optional[int], m: Optional[int], y: Optional[int]) -> str:
    dd = f"{d:02d}" if isinstance(d, int) else ""
    mm = f"{m:02d}" if isinstance(m, int) else ""
    yy = f"{y:04d}" if isinstance(y, int) else ""
    return "-".join([dd, mm, yy])

# -----------------------------
# Calendar primitives
# -----------------------------

def _j_to_g(jy: int, jm: int, jday: int) -> GDate:
    return jd.date(jy, jm, jday).togregorian()

def _g_to_j(g: GDate) -> jd.date:
    return jd.date.fromgregorian(date=g)

def _greg_month_range(y: int, m: int) -> Tuple[GDate, GDate]:
    start = GDate(y, m, 1)
    end = GDate(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1)
    return start, end

def _solar_month_range(y: int, m: int) -> Tuple[GDate, GDate]:
    j_start = jd.date(y, m, 1)
    j_next  = jd.date(y + (m == 12), (m % 12) + 1, 1)
    g_start = j_start.togregorian()
    g_end   = j_next.togregorian() - timedelta(days=1)
    return g_start, g_end

def _solar_year_range(y: int) -> Tuple[GDate, GDate]:
    g_start = jd.date(y, 1, 1).togregorian()
    g_end   = jd.date(y + 1, 1, 1).togregorian() - timedelta(days=1)
    return g_start, g_end

def _greg_year_range(y: int) -> Tuple[GDate, GDate]:
    return GDate(y, 1, 1), GDate(y, 12, 31)

# Approximation helpers (no year)

def _unique_dd_mm_sorted(mmdd: Iterable[Tuple[int, int]]) -> List[str]:
    seen, items = set(), []
    for m, d in mmdd:
        if (m, d) not in seen:
            seen.add((m, d))
            items.append((m, d))
    items.sort()
    return [_fmt_dd_mm(d, m) for m, d in items]

def _approx_day_month_solar_to_greg(day: int, month: int) -> List[str]:
    sample_years = [1398, 1399, 1400, 1401, 1402, 1403, 1404]
    pairs = []
    for y in sample_years:
        try:
            g = _j_to_g(y, month, day)
            pairs.append((g.month, g.day))
        except Exception:
            pass
    opts = _unique_dd_mm_sorted(pairs)
    return opts[:2] if len(opts) > 2 else opts

def _approx_day_month_greg_to_solar(day: int, month: int) -> List[str]:
    sample_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    pairs = []
    for y in sample_years:
        try:
            g = GDate(y, month, day)
            j = _g_to_j(g)
            pairs.append((j.month, j.day))
        except Exception:
            pass
    opts = _unique_dd_mm_sorted(pairs)
    return opts[:2] if len(opts) > 2 else opts

def _approx_month_range_solar_to_greg(month: int) -> Tuple[str, str]:
    sample_years = [1399, 1400, 1401, 1402, 1403, 1404]
    starts, ends = [], []
    for y in sample_years:
        g_start, g_end = _solar_month_range(y, month)
        starts.append((g_start.month, g_start.day))
        ends.append((g_end.month, g_end.day))
    start_opts = _unique_dd_mm_sorted(starts)
    end_opts = _unique_dd_mm_sorted(ends)
    return (start_opts[0], end_opts[-1])

def _approx_month_range_greg_to_solar(month: int) -> Tuple[str, str]:
    sample_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    starts, ends = [], []
    for y in sample_years:
        s, e = _greg_month_range(y, month)
        j_s, j_e = _g_to_j(s), _g_to_j(e)
        starts.append((j_s.month, j_s.day))
        ends.append((j_e.month, j_e.day))
    start_opts = _unique_dd_mm_sorted(starts)
    end_opts = _unique_dd_mm_sorted(ends)
    return (start_opts[0], end_opts[-1])

# ---- NEW: day-only -> list[str] (length 12), dd-mm or ""
def _approx_day_only_list(src: str, tgt: str, day: int) -> List[str]:
    out: List[str] = []
    if src == "solar":
        sample_years = [1398, 1399, 1400, 1401, 1402, 1403, 1404]
        for m in range(1, 13):
            pairs = []
            for y in sample_years:
                try:
                    j = jd.date(y, m, day)  # may raise if invalid
                    if tgt == "gregorian":
                        g = j.togregorian()
                        pairs.append((g.month, g.day))
                    else:
                        pairs.append((m, day))  # solar->solar
                except Exception:
                    pass
            if not pairs:
                out.append("")  # no valid mapping across samples
            else:
                # choose earliest option deterministically
                opts = _unique_dd_mm_sorted(pairs)
                out.append(opts[0])
    else:  # src == gregorian
        sample_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
        for m in range(1, 13):
            pairs = []
            for y in sample_years:
                try:
                    g = GDate(y, m, day)  # may raise if invalid
                    if tgt == "solar":
                        j = _g_to_j(g)
                        pairs.append((j.month, j.day))
                    else:
                        pairs.append((m, day))  # greg->greg
                except Exception:
                    pass
            if not pairs:
                out.append("")
            else:
                opts = _unique_dd_mm_sorted(pairs)
                out.append(opts[0])
    return out

# ---- NEW: day+year -> list[str] (length 12), dd-mm-yyyy or ""
def _day_year_list(src: str, tgt: str, day: int, year: int) -> List[str]:
    out: List[str] = []
    for m in range(1, 13):
        try:
            if src == "solar":
                j = jd.date(year, m, day)
                if tgt == "gregorian":
                    g = j.togregorian()
                    out.append(_fmt_dd_mm_yyyy(g.day, g.month, g.year))
                else:
                    out.append(_fmt_dd_mm_yyyy(j.day, j.month, j.year))
            else:
                g = GDate(year, m, day)
                if tgt == "gregorian":
                    out.append(_fmt_dd_mm_yyyy(g.day, g.month, g.year))
                else:
                    j = _g_to_j(g)
                    out.append(_fmt_dd_mm_yyyy(j.day, j.month, j.year))
        except Exception:
            out.append("")  # invalid day for that month
    return out

# Parse bundle

@dataclass
class ParsedInput:
    day: Optional[int]
    month: Optional[int]
    year: Optional[int]
    parsed: str  # dd-mm-yyyy

def _parse_components(source_calendar: Optional[str], day: Optional[str], month: Optional[str], year: Optional[str]) -> ParsedInput:
    d = _parse_int(day)
    m = _parse_month_token(month, source_calendar)
    if m is None:
        other = "solar" if source_calendar == "gregorian" else "gregorian"
        m = _parse_month_token(month, other)
    y = _parse_int(year)
    return ParsedInput(day=d, month=m, year=y, parsed=_parsed_as_str(d, m, y))

# Public API

def convert_calendar(
    source_calendar: str,
    target_calendar: str,
    day: Optional[str],
    month: Optional[str],
    year: Optional[str],
) -> Dict[str, Any]:
    """
    Convert (day, month, year) from source to target calendar.

    For day-only and day+year, `date` is a list of 12 strings (one per month),
    using "" where the day doesn't exist in that month.
    """
    src = _norm_cal_name(source_calendar)
    tgt = _norm_cal_name(target_calendar)
    if src not in {"solar", "gregorian"} or tgt not in {"solar", "gregorian"}:
        return {
            "kind": "invalid",
            "source_calendar": src or source_calendar,
            "target_calendar": tgt or target_calendar,
            "parsed": "--",
            "date": {"reason": "source/target must be Solar or Gregorian"},
            "standard_format": "dd-mm-yyyy",
        }

    parsed = _parse_components(src, day, month, year)
    d, m, y = parsed.day, parsed.month, parsed.year

    if d is None and m is None and y is None:
        return {
            "kind": "invalid",
            "source_calendar": src,
            "target_calendar": tgt,
            "parsed": parsed.parsed,
            "date": {"reason": "Provide at least one of day, month, or year."},
            "standard_format": "dd-mm-yyyy",
        }

    # FULL DATE → exact
    if d is not None and m is not None and y is not None:
        try:
            if src == "solar":
                g = _j_to_g(y, m, d)
                out = _fmt_dd_mm_yyyy(*( (g.day, g.month, g.year) if tgt == "gregorian" else (d, m, y) ))
            else:
                g = GDate(y, m, d)
                if tgt == "gregorian":
                    out = _fmt_dd_mm_yyyy(g.day, g.month, g.year)
                else:
                    j = _g_to_j(g)
                    out = _fmt_dd_mm_yyyy(j.day, j.month, j.year)
            return {"kind": "exact","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": out,"standard_format": "dd-mm-yyyy"}
        except Exception as e:
            return {"kind": "invalid","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"reason": f"Invalid date: {e}"},"standard_format": "dd-mm-yyyy"}

    # YEAR + MONTH → range
    if d is None and m is not None and y is not None:
        try:
            if src == "solar":
                g_start, g_end = _solar_month_range(y, m)
                if tgt == "gregorian":
                    f, t = _fmt_dd_mm_yyyy(g_start.day, g_start.month, g_start.year), _fmt_dd_mm_yyyy(g_end.day, g_end.month, g_end.year)
                else:
                    j_start, j_end = _g_to_j(g_start), _g_to_j(g_end)
                    f, t = _fmt_dd_mm_yyyy(j_start.day, j_start.month, j_start.year), _fmt_dd_mm_yyyy(j_end.day, j_end.month, j_end.year)
            else:
                g_start, g_end = _greg_month_range(y, m)
                if tgt == "gregorian":
                    f, t = _fmt_dd_mm_yyyy(g_start.day, g_start.month, g_start.year), _fmt_dd_mm_yyyy(g_end.day, g_end.month, g_end.year)
                else:
                    j_start, j_end = _g_to_j(g_start), _g_to_j(g_end)
                    f, t = _fmt_dd_mm_yyyy(j_start.day, j_start.month, j_start.year), _fmt_dd_mm_yyyy(j_end.day, j_end.month, j_end.year)
            return {"kind": "range","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"from": f, "to": t},"standard_format": "dd-mm-yyyy"}
        except Exception as e:
            return {"kind": "invalid","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"reason": f"Invalid month/year: {e}"},"standard_format": "dd-mm-yyyy"}

    # DAY + MONTH (no year) → approx
    if d is not None and m is not None and y is None:
        try:
            if src == "solar":
                opts = _approx_day_month_solar_to_greg(d, m) if tgt == "gregorian" else [_fmt_dd_mm(d, m)]
            else:
                opts = _approx_day_month_greg_to_solar(d, m) if tgt == "solar" else [_fmt_dd_mm(d, m)]
            if not opts:
                return {"kind": "invalid","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"reason": "Day/month invalid in samples."},"standard_format": "dd-mm-yyyy"}
            return {"kind": "approx","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"options": opts[:2]},"standard_format": "dd-mm-yyyy"}
        except Exception as e:
            return {"kind": "invalid","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"reason": f"Invalid day/month: {e}"},"standard_format": "dd-mm-yyyy"}

    # ---- UPDATED: DAY ONLY (no month/year) → list[str]
    if d is not None and m is None and y is None:
        dates = _approx_day_only_list(src, tgt, d)  # ['dd-mm' or '']
        return {
            "kind": "approx",
            "source_calendar": src, "target_calendar": tgt,
            "parsed": parsed.parsed,
            "date": dates,
            "standard_format": "dd-mm-yyyy",
        }

    # ---- UPDATED: DAY + YEAR (no month) → list[str]
    if d is not None and m is None and y is not None:
        dates = _day_year_list(src, tgt, d, y)      # ['dd-mm-yyyy' or '']
        return {
            "kind": "exact",
            "source_calendar": src, "target_calendar": tgt,
            "parsed": parsed.parsed,
            "date": dates,
            "standard_format": "dd-mm-yyyy",
        }

    # MONTH ONLY → range-approx
    if d is None and m is not None and y is None:
        if src == "solar":
            start, end = _approx_month_range_solar_to_greg(m) if tgt == "gregorian" else (_fmt_dd_mm(1, m), _fmt_dd_mm(30, m))
        else:
            last = 31 if m in {1,3,5,7,8,10,12} else (30 if m != 2 else 29)
            start, end = _approx_month_range_greg_to_solar(m) if tgt == "solar" else (_fmt_dd_mm(1, m), _fmt_dd_mm(last, m))
        return {"kind": "range-approx","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"from": start, "to": end},"standard_format": "dd-mm-yyyy"}

    # YEAR ONLY → range
    if d is None and m is None and y is not None:
        if src == "solar":
            g_start, g_end = _solar_year_range(y)
            if tgt == "gregorian":
                f, t = _fmt_dd_mm_yyyy(g_start.day, g_start.month, g_start.year), _fmt_dd_mm_yyyy(g_end.day, g_end.month, g_end.year)
            else:
                j_start, j_end = _g_to_j(g_start), _g_to_j(g_end)
                f, t = _fmt_dd_mm_yyyy(j_start.day, j_start.month, j_start.year), _fmt_dd_mm_yyyy(j_end.day, j_end.month, j_end.year)
        else:
            g_start, g_end = _greg_year_range(y)
            if tgt == "gregorian":
                f, t = _fmt_dd_mm_yyyy(g_start.day, g_start.month, g_start.year), _fmt_dd_mm_yyyy(g_end.day, g_end.month, g_end.year)
            else:
                j_start, j_end = _g_to_j(g_start), _g_to_j(g_end)
                f, t = _fmt_dd_mm_yyyy(j_start.day, j_start.month, j_start.year), _fmt_dd_mm_yyyy(j_end.day, j_end.month, j_end.year)
        return {"kind": "range","source_calendar": src,"target_calendar": tgt,"parsed": parsed.parsed,"date": {"from": f, "to": t},"standard_format": "dd-mm-yyyy"}

    return {
        "kind": "invalid",
        "source_calendar": src, "target_calendar": tgt,
        "parsed": parsed.parsed,
        "date": {"reason": "Unsupported or ambiguous combination."},
        "standard_format": "dd-mm-yyyy",
    }



if __name__ == "__main__":
    import json

    def run_case(src, tgt, d, m, y, label=None):
        inp = {
            "source_calendar": src,
            "target_calendar": tgt,
            "day": d,
            "month": m,
            "year": y,
        }
        out = convert_calendar(src, tgt, d, m, y)

        def summarize(result):
            kind = result.get("kind")
            date = result.get("date")
            if kind == "exact" and isinstance(date, str):
                return date
            if kind in ("range", "range-approx") and isinstance(date, dict):
                return f'{date.get("from","?")} → {date.get("to","?")}'
            if kind == "approx" and isinstance(date, dict):
                opts = date.get("options", [])
                return ", ".join(opts) if opts else "(no options)"
            if kind == "invalid" and isinstance(date, dict):
                return f'Invalid: {date.get("reason","")}'
            return ""

        print("=" * 88)
        if label:
            print(f"TEST: {label}")
        print("- Input:")
        print(json.dumps(inp, ensure_ascii=False, indent=2))
        print("- Output:")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"- Summary: {summarize(out)}")

    TESTS = [
        ("solar", "gregorian", "12", "Mehr", "1404", "Full date Solar→Greg (expect 04-10-2025)"),
        ("solar", "gregorian", "", "Mehr", "1404", "Month+Year Solar→Greg (expect 23-09-2025 → 22-10-2025)"),
        ("solar", "gregorian", "12", "Mehr", "", "Day+Month approx Solar→Greg"),
        ("solar", "gregorian", "", "Mehr", "", "Month-only range-approx Solar→Greg (expect 23-09 → 22-10)"),
        ("solar", "gregorian", "", "", "1404", "Year-only Solar→Greg (expect 21-03-2025 → 20-03-2026)"),
        ("میلادی", "solar", "04", "Oct", "2025", "greg→solar with Persian source label"),
        ("solar", "gregorian", "۱۲", "مهر", "۱۴۰۴", "Persian digits+names Solar→Greg"),
        ("gregorian", "gregorian", "4", "october", "2025", "greg→greg passthrough"),
        ("solar", "solar", "12", "Mehr", "1404", "solar→solar passthrough"),
        ("gregorian", "solar", "04", "October", "2025", "Full date Greg→Solar (should be 12-07-1404)"),
        ("gregorian", "solar", "29", "February", "2024", "Leap day Greg→Solar"),
        ("gregorian", "solar", "", "February", "", "Month-only Greg→Solar (range-approx)"),
        ("gregorian", "gregorian", "", "Aug", "2025", "Month+Year Greg→Greg range"),
        ("solar", "gregorian", "", "", "۱۴۰۳", "Year-only Solar→Greg with Persian digits"),
        ("solar", "gregorian", "5", "", "", "Day-only Solar→Greg (approx; 12 per-month options)"),
        ("lunar", "gregorian", "1", "jan", "2025", "Invalid calendar name"),
        ("gregorian", "solar", "٠٤", "اکتبر", "٢٠٢٥", "Arabic-Indic digits + Persian Greg month"),
        ("solar", "gregorian", "1", "day", "1404", "Alt spelling 'day' for Dey (Solar month 10)"),
        ("gregorian", "solar", "07", "Sept", "2025", "English abbrev 'Sept' Greg→Solar"),
        ("gregorian", "solar", "۱۵", "ژوئیه", "۲۰۲۵", "Persian Greg month name + Persian digits"),
        ("solar", "gregorian", "30", "Esfand", "1403", "Edge near Solar year end (validity depends on leap year)"),
        ("solar", "gregorian", "", "Shahrivar", "1404", "Solar month name (EN) → Greg range"),
        ("solar", "solar", "", "بهمن", "", "Solar month-only Solar→Solar (range-approx in Solar)"),
        ("gregorian", "gregorian", "", "", "2025", "Year-only Greg→Greg (exact range)"),
        ("solar", "gregorian", "12", "", "", "Day-only Solar→Greg (approx; Mehr typically ~ early Oct)"),
        ("gregorian", "solar", "31", "", "", "Day-only Greg→Solar (approx; invalid in some months)"),
        ("gregorian", "gregorian", "31", "", "", "Day-only Greg→Greg (approx; April/June/Sep/Nov invalid)"),
        ("solar", "solar", "30", "", "", "Day-only Solar→Solar (approx; Esfand may be invalid)"),
        ("solar", "gregorian", "12", "", "1404", "Day+Year Solar→Greg (per-month exact; Mehr→04-10-2025)"),
        ("gregorian", "solar", "29", "", "2024", "Day+Year Greg→Solar (per-month exact; Feb valid in leap year)"),
        ("gregorian", "gregorian", "15", "", "2025", "Day+Year Greg→Greg (per-month exact list)"),
        ("solar", "solar", "15", "", "1404", "Day+Year Solar→Solar (per-month exact list)"),
    ]


    for idx, (src, tgt, d, m, y, label) in enumerate(TESTS, 1):
        run_case(src, tgt, d, m, y, f"#{idx} - {label}")

    print("=" * 88)
    print(f"Total tests run: {len(TESTS)}")
