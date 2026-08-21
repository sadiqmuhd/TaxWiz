"""The tax engine is pure arithmetic, so these assert real numbers, not mocks."""

import pytest

import tax


# ── PAYE ───────────────────────────────────────────────────────────────────────

def test_paye_worked_example():
    """N1.2M gross, N96k pension, N30k NHF, checked band by band by hand.

    CRA        = max(200,000, 1% of 1.2M) + 20% of 1.2M = 200,000 + 240,000
    reliefs    = 440,000 + 96,000 + 30,000              = 566,000
    taxable    = 1,200,000 - 566,000                    = 634,000
    tax        = 300,000 @ 7%  =  21,000
               + 300,000 @ 11% =  33,000
               +  34,000 @ 15% =   5,100
                                 --------
                                  59,100
    """
    result = tax.calculate_paye({"gross_income": 1_200_000, "pension": 96_000, "nhf": 30_000})

    assert result["consolidated_relief_allowance"] == 440_000
    assert result["total_relief"] == 566_000
    assert result["taxable_income"] == 634_000
    assert result["computed_tax"] == 59_100
    assert result["annual_tax"] == 59_100
    assert result["monthly_tax"] == 4_925
    assert result["minimum_tax_applied"] is False
    assert result["effective_rate"] == pytest.approx(4.93, abs=0.01)


def test_paye_minimum_tax_applies_to_low_earners():
    """At N250k the reliefs (200,000 + 20%) wipe out taxable income entirely,
    so the 1% minimum tax governs instead of the band computation."""
    result = tax.calculate_paye({"gross_income": 250_000})

    assert result["total_relief"] == 250_000
    assert result["taxable_income"] == 0
    assert result["computed_tax"] == 0
    assert result["minimum_tax"] == 2_500
    assert result["annual_tax"] == 2_500
    assert result["minimum_tax_applied"] is True


def test_paye_reliefs_never_exceed_gross_income():
    """Huge pension contributions must not produce negative taxable income."""
    result = tax.calculate_paye({"gross_income": 400_000, "pension": 900_000})

    assert result["total_relief"] == 400_000
    assert result["taxable_income"] == 0
    assert result["annual_tax"] == 4_000  # the 1% floor


def test_paye_top_band_is_open_ended():
    """A very high earner is taxed at 24% on everything above N3.2M."""
    result = tax.calculate_paye({"gross_income": 20_000_000})

    top_band = result["bands"][-1]
    assert top_band["rate"] == 24
    assert top_band["applied"] is True
    # Bands are consumed in order and never taxed twice.
    assert sum(band["amount_taxed"] for band in result["bands"]) == pytest.approx(
        result["taxable_income"]
    )


# ── CIT ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("turnover,expected_rate", [
    (10_000_000, 0),    # small
    (25_000_000, 0),    # boundary: still small
    (25_000_001, 20),   # medium
    (100_000_000, 20),  # boundary: still medium
    (100_000_001, 30),  # large
])
def test_cit_rate_bands(turnover, expected_rate):
    result = tax.calculate_cit({"turnover": turnover, "assessable_profit": 5_000_000})
    assert result["rate"] == expected_rate


def test_cit_deducts_capital_allowances():
    result = tax.calculate_cit({
        "turnover": 500_000_000,
        "assessable_profit": 12_000_000,
        "capital_allowances": 2_000_000,
    })
    assert result["taxable_profit"] == 10_000_000
    assert result["tax_payable"] == 3_000_000  # 30%


def test_cit_allowances_cannot_create_negative_tax():
    result = tax.calculate_cit({
        "turnover": 500_000_000,
        "assessable_profit": 1_000_000,
        "capital_allowances": 9_000_000,
    })
    assert result["taxable_profit"] == 0
    assert result["tax_payable"] == 0


# ── VAT ────────────────────────────────────────────────────────────────────────

def test_vat_exclusive_adds_seven_and_a_half_percent():
    result = tax.calculate_vat({"amount": 500_000, "mode": "exclusive"})
    assert result["net_amount"] == 500_000
    assert result["vat_amount"] == 37_500
    assert result["total_amount"] == 537_500


def test_vat_inclusive_backs_the_tax_out_of_the_total():
    result = tax.calculate_vat({"amount": 537_500, "mode": "inclusive"})
    assert result["net_amount"] == 500_000
    assert result["vat_amount"] == 37_500
    assert result["total_amount"] == 537_500


def test_vat_defaults_to_exclusive():
    assert tax.calculate_vat({"amount": 1_000})["mode"] == "exclusive"


def test_vat_rejects_unknown_mode():
    with pytest.raises(tax.TaxInputError):
        tax.calculate_vat({"amount": 1_000, "mode": "reverse"})


# ── WHT ────────────────────────────────────────────────────────────────────────

def test_wht_five_percent_category():
    result = tax.calculate_wht({"amount": 1_000_000, "transaction_type": "consultancy"})
    assert result["rate"] == 5
    assert result["wht_deducted"] == 50_000
    assert result["net_amount"] == 950_000


def test_wht_ten_percent_category():
    result = tax.calculate_wht({"amount": 1_000_000, "transaction_type": "dividends"})
    assert result["rate"] == 10
    assert result["wht_deducted"] == 100_000


def test_wht_rejects_unknown_transaction_type():
    with pytest.raises(tax.TaxInputError) as excinfo:
        tax.calculate_wht({"amount": 1_000, "transaction_type": "bribe"})
    assert "transaction_type" in str(excinfo.value)


# ── Input validation ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    {},                                    # missing required field
    {"gross_income": -1},                  # negative
    {"gross_income": 0},                   # zero
    {"gross_income": "not a number"},      # non-numeric
    {"gross_income": None},                # null
    {"gross_income": 100_000, "pension": -5},
])
def test_paye_rejects_bad_input(payload):
    with pytest.raises(tax.TaxInputError):
        tax.calculate_paye(payload)


def test_numeric_strings_are_accepted():
    """The browser sends form values as strings."""
    assert tax.calculate_vat({"amount": "1000"})["vat_amount"] == 75


def test_unknown_tax_type_is_rejected():
    with pytest.raises(tax.TaxInputError):
        tax.calculate("capital-gains", {"amount": 1})


def test_every_registered_calculator_is_reachable():
    assert set(tax.CALCULATORS) == {"paye", "cit", "vat", "wht"}
