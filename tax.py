"""Deterministic Nigerian tax calculations.

Every function here is pure: same inputs -> same outputs, no network calls, no
LLM involvement. The language model in TaxWiz never performs arithmetic; it only
answers questions about the law. Anything numeric goes through this module.

Source of the rules: the Nigeria Tax Act, 2025 (Act No. 7, Official Gazette
No. 117, Vol. 112), commencing 1 January 2026, as published by the National
Assembly ("Edited 1-1-2026" edition). Section references are to that Act and
were checked against its text, not taken from memory or secondary summaries.

Deliberately NOT implemented, because the Act does not supply the inputs:
  * the exemption for employees earning the national minimum wage or less
    (s. 58, s. 162(1)(t)) — the Act does not state the wage amount;
  * the 15% minimum effective tax rate (s. 57) — it is computed from audited
    accounts, and applies only to large MNE-group members or companies with
    turnover of N50bn or more (flagged in the result, not computed);
  * the reduction of the company rate to 25% — it takes effect only from a
    date set by presidential order (s. 56 proviso).

Withholding tax rates are not set by this Act: it refers to "the rate prescribed
by Nigeria Tax Administration Act". WHT_RATES below is therefore unverified.
"""

from typing import Any, Dict, List, Optional, Tuple

# ── Personal income tax (PAYE) ─────────────────────────────────────────────────

# Fourth Schedule: (width of band, rate, label). The final band is open-ended.
PAYE_BANDS: List[Tuple[float, float, str]] = [
    (800_000, 0.00, "First ₦800,000"),
    (2_200_000, 0.15, "Next ₦2,200,000"),
    (9_000_000, 0.18, "Next ₦9,000,000"),
    (13_000_000, 0.21, "Next ₦13,000,000"),
    (25_000_000, 0.23, "Next ₦25,000,000"),
    (float("inf"), 0.25, "Above ₦50,000,000"),
]

# s. 30(2)(a)(vi): rent relief of 20% of annual rent paid, at most N500,000.
RENT_RELIEF_RATE = 0.20
RENT_RELIEF_CAP = 500_000

# s. 30(2)(a): eligible deductions taken at face value, in the order listed there.
DEDUCTIONS: List[Tuple[str, str]] = [
    ("nhf", "National Housing Fund contributions"),
    ("nhis", "National Health Insurance contributions"),
    ("pension", "Pension contributions"),
    ("mortgage_interest", "Interest on an owner-occupied home loan"),
    ("life_assurance", "Life assurance / annuity premiums"),
]

# ── VAT ────────────────────────────────────────────────────────────────────────

VAT_RATE = 0.075  # s. 147

# ── Company income tax and development levy ────────────────────────────────────

# Definition of "small company": gross turnover of N100m or less per annum AND
# total fixed assets not exceeding N250m. Both tests must be met.
SMALL_COMPANY_TURNOVER = 100_000_000
SMALL_COMPANY_FIXED_ASSETS = 250_000_000
CIT_SMALL_RATE = 0.0   # s. 56(a)
CIT_RATE = 0.30        # s. 56(b)
DEVELOPMENT_LEVY_RATE = 0.04  # s. 59(1): on assessable profits, not small companies
MINIMUM_ETR_TURNOVER = 50_000_000_000  # s. 57(2)(b)

# ── Withholding tax (unverified: prescribed under the Tax Administration Act) ──

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
    """Annual personal income tax under the Nigeria Tax Act 2025.

    Chargeable income is total income less the eligible deductions of s. 30,
    including rent relief; the Fourth Schedule bands then apply. There is no
    consolidated relief allowance and no minimum tax for individuals in the Act.
    """
    gross = _amount(payload, "gross_income", required=True)
    if gross <= 0:
        raise TaxInputError("'gross_income' must be greater than zero.")

    deductions = {key: _amount(payload, key) for key, _ in DEDUCTIONS}
    annual_rent = _amount(payload, "annual_rent")
    rent_relief = min(annual_rent * RENT_RELIEF_RATE, RENT_RELIEF_CAP)

    claimed = sum(deductions.values()) + rent_relief
    eligible = min(claimed, gross)          # deductions cannot create a loss
    chargeable = gross - eligible

    remaining = chargeable
    annual_tax = 0.0
    bands = []
    for width, rate, label in PAYE_BANDS:
        taxed_here = min(remaining, width) if remaining > 0 else 0.0
        tax_here = taxed_here * rate
        annual_tax += tax_here
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

    result = {"tax_type": "paye", "gross_income": _round(gross)}
    result.update({key: _round(value) for key, value in deductions.items()})
    result.update({
        "annual_rent": _round(annual_rent),
        "rent_relief": _round(rent_relief),
        "eligible_deductions": _round(eligible),
        "chargeable_income": _round(chargeable),
        "annual_tax": _round(annual_tax),
        "monthly_tax": _round(annual_tax / 12),
        "effective_rate": round(annual_tax / gross * 100, 2),
        "bands": bands,
    })
    return result


# ── Company income tax ─────────────────────────────────────────────────────────

def calculate_cit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Company income tax (s. 56) and development levy (s. 59) for a
    Nigerian-resident company.

    Whether the company is small depends on turnover AND fixed assets, so
    `fixed_assets` is required whenever turnover alone does not settle it.
    """
    turnover = _amount(payload, "turnover", required=True)
    profit = _amount(payload, "assessable_profit", required=True)
    allowances = _amount(payload, "capital_allowances")

    if turnover <= 0:
        raise TaxInputError("'turnover' must be greater than zero.")

    fixed_assets: Optional[float] = None
    if turnover <= SMALL_COMPANY_TURNOVER:
        if payload.get("fixed_assets") in (None, ""):
            raise TaxInputError(
                "'fixed_assets' is required when turnover is N100,000,000 or less: a "
                "company is small only if its fixed assets are also N250,000,000 or less."
            )
        fixed_assets = _amount(payload, "fixed_assets", required=True)
    elif payload.get("fixed_assets") not in (None, ""):
        fixed_assets = _amount(payload, "fixed_assets")

    is_small = (turnover <= SMALL_COMPANY_TURNOVER
                and fixed_assets is not None
                and fixed_assets <= SMALL_COMPANY_FIXED_ASSETS)

    if is_small:
        rate, category = CIT_SMALL_RATE, "Small company"
    else:
        rate, category = CIT_RATE, "Not a small company"

    taxable_profit = max(0.0, profit - allowances)
    cit = taxable_profit * rate
    levy_rate = 0.0 if is_small else DEVELOPMENT_LEVY_RATE
    levy = profit * levy_rate   # s. 59 charges the levy on assessable profits

    return {
        "tax_type": "cit",
        "turnover": _round(turnover),
        "fixed_assets": None if fixed_assets is None else _round(fixed_assets),
        "category": category,
        "is_small_company": is_small,
        "rate": round(rate * 100, 2),
        "assessable_profit": _round(profit),
        "capital_allowances": _round(allowances),
        "taxable_profit": _round(taxable_profit),
        "tax_payable": _round(cit),
        "development_levy_rate": round(levy_rate * 100, 2),
        "development_levy": _round(levy),
        "total_payable": _round(cit + levy),
        "minimum_etr_may_apply": turnover >= MINIMUM_ETR_TURNOVER,
    }


# ── VAT ────────────────────────────────────────────────────────────────────────

def calculate_vat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """VAT at 7.5% (s. 147), on an amount that excludes or includes VAT."""
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
    """Withholding tax deducted at source, by transaction category.

    The rates are not in the Nigeria Tax Act 2025 and have not been verified.
    """
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
