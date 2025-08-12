import jdatetime
import calendar
from datetime import date as py_date, datetime as py_datetime

# ──────────────────────────  DATE / CALENDAR HELPERS  ───────────────────── #

CAL_GREGORIAN = "Gregorian"
CAL_SOLAR = "Solar"  # Solar Hijri (Shamsi / Jalali)

# Persian and Arabic-Indic digits → Latin
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_eastern_digits(s: str) -> str:
    """Convert Persian/Arabic-Indic numerals in a string to ASCII digits."""
    return s.translate(_PERSIAN_DIGITS).translate(_ARABIC_INDIC_DIGITS)


# Month name maps (minimal but practical set)
GREG_MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Solar Hijri months (Persian script + common Latin transliterations)
JALALI_MONTHS = {
    "farvardin": 1,     "فروردین": 1,
    "ordibehesht": 2,   "اردیبهشت": 2,
    "khordad": 3,       "خرداد": 3,
    "tir": 4,           "تیر": 4,
    "mordad": 5,        "مرداد": 5,
    "shahrivar": 6,     "شهریور": 6,
    "mehr": 7,          "مهر": 7,
    "aban": 8,          "آبان": 8,
    "azar": 9,          "آذر": 9,
    "dey": 10,          "دی": 10,
    "bahman": 11,       "بهمن": 11,
    "esfand": 12,       "اسفند": 12,
}

def jalali_days_in_month(year: int, month: int) -> int:
    """Number of days in a given Jalali month (uses jdatetime leap logic)."""
    if month < 1 or month > 12:
        raise ValueError("Invalid Jalali month")
    if month <= 6:
        return 31
    if 7 <= month <= 11:
        return 30
    # Esfand (12)
    # jdatetime leap: create a date and call isleap()
    if jdatetime is None:
        # conservative default if lib missing
        return 29
    leap = jdatetime.date(year, 1, 1).isleap()
    return 30 if leap else 29

def gregorian_days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _ensure_jdatetime():
    if jdatetime is None:
        raise RuntimeError("jdatetime is not installed. Please: pip install jdatetime")


def jalali_to_gregorian(y: int, m: int, d: int) -> py_date:
    """Exact conversion Jalali → Gregorian date."""
    _ensure_jdatetime()
    return jdatetime.date(y, m, d).togregorian()

def gregorian_to_jalali(y: int, m: int, d: int) -> jdatetime.date:
    """Exact conversion Gregorian → Jalali date."""
    _ensure_jdatetime()
    return jdatetime.date.fromgregorian(year=y, month=m, day=d)

def format_iso(y: int, m: int | None = None, d: int | None = None) -> str:
    """ISO-ish string with available granularity."""
    if m is None:
        return f"{y:04d}"
    if d is None:
        return f"{y:04d}-{m:02d}"
    return f"{y:04d}-{m:02d}-{d:02d}"