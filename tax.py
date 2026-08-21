"""Deterministic Nigerian tax calculations.

Every function here is pure: same inputs -> same outputs, no network calls, no
LLM involvement. The language model in TaxWiz never performs arithmetic; it only
answers questions about the law. Anything numeric goes through this module.

The rates and reliefs implemented here mirror the schedule the project shipped
with: the graduated personal income tax bands and reliefs, 7.5% VAT, and the
turnover-banded company income tax rates. See README "Limitations".
"""

from typing import Any, Dict, List, Tuple

# ── Rate tables ────────────────────────────────────────────────────────────────

# (width of band, rate, human label). The final band is open-ended.
PAYE_BANDS: List[Tuple[float, float, str]] = [
    (300_000, 0.07, "First ₦300,000"),
    (300_000, 0.11, "Next ₦300,000"),
    (500_000, 0.15, "Next ₦500,000"),
    (500_000, 0.19, "Next ₦500,000"),
    (1_600_000, 0.21, "Next ₦1,600,000"),
    (float("inf"), 0.24, "Above ₦3,200,000"),
]

CRA_FLOOR = 200_000           # fixed part of the Consolidated Relief Allowance...
CRA_FIXED_RATE = 0.01         # ...or 1% of gross income, whichever is higher
CRA_VARIABLE_RATE = 0.20      # plus 20% of gross income
MINIMUM_TAX_RATE = 0.01       # 1% of gross income floor

VAT_RATE = 0.075

CIT_SMALL_TURNOVER = 25_000_000
CIT_MEDIUM_TURNOVER = 100_000_000
CIT_SMALL_RATE = 0.0
CIT_MEDIUM_RATE = 0.20
CIT_LARGE_RATE = 0.30

WHT_RATES: Dict[str, Dict[str, Any]] = {
    "dividends":    {"rate": 0.10, "label": "Dividends / Interest / Rent"},
    "director":     {"rate": 0.10, "label": "Director fees"},
    "commission":   {"rate": 0.10, "label": "Commission / Royalties"},
    "construction": {"rate": 0.05, "label": "Construction / Survey"},
    "consultancy":  {"rate": 0.05, "label": "Consultancy / Management fees"},
    "professional": {"rate": 0.05, "label": "Technical / Professional fees"},
    "contracts":    {"rate": 0.05, "label": "Contracts — goods & services"},
}


class TaxInputError(ValueError):
    """Raised when supplied figures are missing, non-numeric or negative."""


# ── Input helpers ──────────────────────────────────────────────────────────────

def _amount(payload: Dict[str, Any], field: str, required: bool = False) -> float:
    """Read a non-negative, finite number out of the request payload."""
    raw = payload.get(field)
    if raw is None or raw == "":
        if required:
            raise TaxInputError("'%s' is required." % field)
        return 0.0
    if isinstance(raw, bool):
        raise TaxInputError("'%s' must be a number." % field)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise TaxInputError("'%s' must be a number." % field)
    if value != value or value in (float("inf"), float("-inf")):
        raise TaxInputError("'%s' must be a finite number." % field)
    if value < 0:
        raise TaxInputError("'%s' cannot be negative." % field)
    return value


def _round(value: float) -> float:
    return round(value, 2)


# ── PAYE (personal income tax) ─────────────────────────────────────────────────

def calculate_paye(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Annual PAYE on employment income, using the graduated band schedule.

    Reliefs applied: the Consolidated Relief Allowance (higher of N200,000 or 1%
    of gross, plus 20% of gross), pension contributions and NHF contributions.
    A minimum tax of 1% of gross income applies where the banded computation
    produces a smaller figure.
    """
    gross = _amount(payload, "gross_income", required=True)
    if gross <= 0:
        raise TaxInputError("'gross_income' must be greater than zero.")
    pension = _amount(payload, "pension")
    nhf = _amount(payload, "nhf")

    cra = max(CRA_FLOOR, gross * CRA_FIXED_RATE) + gross * CRA_VARIABLE_RATE
    total_relief = min(cra + pension + nhf, gross)
    taxable_income = max(0.0, gross - total_relief)

    remaining = taxable_income
    computed_tax = 0.0
    bands = []
    for width, rate, label in PAYE_BANDS:
        taxed_here = min(remaining, width) if remaining > 0 else 0.0
        tax_here = taxed_here * rate
        computed_tax += tax_here
        remaining -= taxed_here
        bands.append({
            "label": label,
            "rate": round(rate * 100, 2),
            "amount_taxed": _round(taxed_here),
            "tax": _round(tax_here),
            "applied": taxed_here > 0,
        })
        if remaining <= 0:
            break

    minimum_tax = gross * MINIMUM_TAX_RATE
    annual_tax = max(computed_tax, minimum_tax)

    return {
        "tax_type": "paye",
        "gross_income": _round(gross),
        "pension": _round(pension),
        "nhf": _round(nhf),
        "consolidated_relief_allowance": _round(cra),
        "total_relief": _round(total_relief),
        "taxable_income": _round(taxable_income),
        "computed_tax": _round(computed_tax),
        "minimum_tax": _round(minimum_tax),
        "minimum_tax_applied": minimum_tax > computed_tax,
        "annual_tax": _round(annual_tax),
        "monthly_tax": _round(annual_tax / 12),
        "effective_rate": round(annual_tax / gross * 100, 2),
        "bands": bands,
    }


# ── Company income tax ─────────────────────────────────────────────────────────

def calculate_cit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Company income tax, banded by annual turnover."""
    turnover = _amount(payload, "turnover", required=True)
    profit = _amount(payload, "assessable_profit", required=True)
    allowances = _amount(payload, "capital_allowances")

    if turnover <= 0:
        raise TaxInputError("'turnover' must be greater than zero.")

    if turnover <= CIT_SMALL_TURNOVER:
        rate = CIT_SMALL_RATE
        category = "Small company (≤ ₦25M turnover)"
    elif turnover <= CIT_MEDIUM_TURNOVER:
        rate = CIT_MEDIUM_RATE
        category = "Medium company (₦25M – ₦100M turnover)"
    else:
        rate = CIT_LARGE_RATE
        category = "Large company (> ₦100M turnover)"

    taxable_profit = max(0.0, profit - allowances)
    tax = taxable_profit * rate

    return {
        "tax_type": "cit",
        "turnover": _round(turnover),
        "category": category,
        "rate": round(rate * 100, 2),
        "assessable_profit": _round(profit),
        "capital_allowances": _round(allowances),
        "taxable_profit": _round(taxable_profit),
        "tax_payable": _round(tax),
    }


# ── VAT ────────────────────────────────────────────────────────────────────────

def calculate_vat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """VAT at 7.5%, on an amount that either excludes or already includes VAT."""
    amount = _amount(payload, "amount", required=True)
    if amount <= 0:
        raise TaxInputError("'amount' must be greater than zero.")

    mode = str(payload.get("mode", "exclusive")).strip().lower()
    if mode not in ("exclusive", "inclusive"):
        raise TaxInputError("'mode' must be either 'exclusive' or 'inclusive'.")

    if mode == "exclusive":
        net = amount
        vat = amount * VAT_RATE
        total = net + vat
    else:
        total = amount
        net = amount / (1 + VAT_RATE)
        vat = total - net

    return {
        "tax_type": "vat",
        "mode": mode,
        "rate": round(VAT_RATE * 100, 2),
        "net_amount": _round(net),
        "vat_amount": _round(vat),
        "total_amount": _round(total),
    }


# ── Withholding tax ────────────────────────────────────────────────────────────

def calculate_wht(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Withholding tax deducted at source, by transaction category."""
    amount = _amount(payload, "amount", required=True)
    if amount <= 0:
        raise TaxInputError("'amount' must be greater than zero.")

    key = str(payload.get("transaction_type", "")).strip().lower()
    if key not in WHT_RATES:
        raise TaxInputError(
            "'transaction_type' must be one of: " + ", ".join(sorted(WHT_RATES))
        )

    rate = WHT_RATES[key]["rate"]
    wht = amount * rate

    return {
        "tax_type": "wht",
        "transaction_type": key,
        "transaction_label": WHT_RATES[key]["label"],
        "gross_amount": _round(amount),
        "rate": round(rate * 100, 2),
        "wht_deducted": _round(wht),
        "net_amount": _round(amount - wht),
    }


CALCULATORS = {
    "paye": calculate_paye,
    "cit": calculate_cit,
    "vat": calculate_vat,
    "wht": calculate_wht,
}


def calculate(tax_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the calculator registered for `tax_type`."""
    key = str(tax_type or "").strip().lower()
    if key not in CALCULATORS:
        raise TaxInputError(
            "Unknown tax type '%s'. Supported: %s" % (tax_type, ", ".join(CALCULATORS))
        )
    if not isinstance(payload, dict):
        raise TaxInputError("Request body must be a JSON object.")
    return CALCULATORS[key](payload)
