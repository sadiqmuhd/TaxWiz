"""The tax engine is pure arithmetic, so these assert real numbers, not mocks.

PAYE and CIT expectations are worked by hand from the Nigeria Tax Act 2025
(Fourth Schedule, ss. 30, 56, 59 and the definition of "small company"), and
the working is written out so a reviewer can check it without the code.
"""

import pytest

import tax


# ── PAYE: the Fourth Schedule bands ────────────────────────────────────────────

def test_paye_bands_match_the_fourth_schedule():
    """First N800,000 at 0%; next N2.2m at 15%; next N9m at 18%; next N13m at
    21%; next N25m at 23%; above N50m at 25%."""
    assert [(w, r) for w, r, _ in tax.PAYE_BANDS] == [
        (800_000, 0.00), (2_200_000, 0.15), (9_000_000, 0.18),
        (13_000_000, 0.21), (25_000_000, 0.23), (float("inf"), 0.25),
    ]
    # The schedule's "Above N50,000,000" must be where the finite bands end.
    assert sum(w for w, _, _ in tax.PAYE_BANDS[:-1]) == 50_000_000


def test_paye_worked_example_with_no_deductions():
    """N6,000,000 a year (N500,000 a month), no deductions claimed.

    chargeable = 6,000,000
    tax        =   800,000 @ 0%  =       0
               + 2,200,000 @ 15% = 330,000
               + 3,000,000 @ 18% = 540,000
                                   -------
                                   870,000
    """
    result = tax.calculate_paye({"gross_income": 6_000_000})

    assert result["eligible_deductions"] == 0
    assert result["chargeable_income"] == 6_000_000
    assert result["annual_tax"] == 870_000
    assert result["monthly_tax"] == 72_500
    assert result["effective_rate"] == 14.5


def test_paye_worked_example_with_pension_and_nhf():
    """N1.2m gross, N96,000 pension, N30,000 NHF (s. 30 eligible deductions).

    chargeable = 1,200,000 - 96,000 - 30,000 = 1,074,000
    tax        = 800,000 @ 0% + 274,000 @ 15% = 41,100
    """
    result = tax.calculate_paye({"gross_income": 1_200_000, "pension": 96_000, "nhf": 30_000})

    assert result["eligible_deductions"] == 126_000
    assert result["chargeable_income"] == 1_074_000
    assert result["annual_tax"] == 41_100


@pytest.mark.parametrize("annual_rent,relief,expected_tax", [
    # chargeable = 6,000,000 - relief; tax = 330,000 + (chargeable - 3,000,000) x 18%
    (1_000_000, 200_000, 834_000),   # 20% of rent = 200,000 -> 5.8m -> 330,000 + 504,000
    (2_500_000, 500_000, 780_000),   # 20% is exactly the 500,000 cap
    (3_000_000, 500_000, 780_000),   # 20% = 600,000, capped at 500,000 -> 5.5m
])
def test_rent_relief_is_20_percent_capped_at_500k(annual_rent, relief, expected_tax):
    """s. 30(2)(a)(vi): 20% of annual rent paid, subject to a maximum of N500,000."""
    result = tax.calculate_paye({"gross_income": 6_000_000, "annual_rent": annual_rent})

    assert result["rent_relief"] == relief
    assert result["annual_tax"] == expected_tax


def test_every_section_30_deduction_is_subtracted():
    result = tax.calculate_paye({
        "gross_income": 10_000_000, "nhf": 100_000, "nhis": 50_000, "pension": 800_000,
        "mortgage_interest": 300_000, "life_assurance": 150_000, "annual_rent": 1_000_000,
    })

    # 100,000 + 50,000 + 800,000 + 300,000 + 150,000 + 200,000 (rent relief)
    assert result["eligible_deductions"] == 1_600_000
    assert result["chargeable_income"] == 8_400_000


@pytest.mark.parametrize("gross,expected", [
    (600_000, 0),               # wholly inside the 0% band
    (800_000, 0),               # the 0% band's upper edge
    (800_100, 15),              # 100 into the 15% band
    (50_000_000, 10_430_000),   # 0 + 330,000 + 1,620,000 + 2,730,000 + 5,750,000
    (100_000_000, 22_930_000),  # the above + 50,000,000 @ 25% = 12,500,000
])
def test_paye_band_boundaries(gross, expected):
    assert tax.calculate_paye({"gross_income": gross})["annual_tax"] == expected


def test_there_is_no_minimum_tax_for_individuals():
    """The old 1%-of-gross minimum tax is not in the 2025 Act."""
    result = tax.calculate_paye({"gross_income": 500_000})

    assert result["annual_tax"] == 0
    assert "minimum_tax" not in result


def test_deductions_never_exceed_gross_income():
    result = tax.calculate_paye({"gross_income": 400_000, "pension": 900_000})

    assert result["eligible_deductions"] == 400_000
    assert result["chargeable_income"] == 0
    assert result["annual_tax"] == 0


def test_bands_are_consumed_in_order_and_never_twice():
    result = tax.calculate_paye({"gross_income": 80_000_000})

    assert result["bands"][-1]["rate"] == 25
    assert result["bands"][-1]["applied"] is True
    assert sum(b["amount_taxed"] for b in result["bands"]) == pytest.approx(
        result["chargeable_income"])


# ── CIT (s. 56), the small-company test, and the development levy (s. 59) ──────

@pytest.mark.parametrize("turnover,fixed_assets,small", [
    (50_000_000, 30_000_000, True),
    (100_000_000, 250_000_000, True),     # both limits are inclusive
    (100_000_000, 250_000_001, False),    # assets one naira over
    (100_000_001, 0, False),              # turnover one naira over
    (10_000_000, 400_000_000, False),     # tiny turnover, large assets
])
def test_small_company_needs_both_tests(turnover, fixed_assets, small):
    """'Small company': turnover of N100m or less AND fixed assets of N250m or less."""
    result = tax.calculate_cit({"turnover": turnover, "fixed_assets": fixed_assets,
                                "assessable_profit": 5_000_000})

    assert result["is_small_company"] is small
    assert result["rate"] == (0 if small else 30)


def test_small_company_pays_no_cit_and_no_levy():
    result = tax.calculate_cit({"turnover": 50_000_000, "fixed_assets": 30_000_000,
                                "assessable_profit": 12_000_000})

    assert result["tax_payable"] == 0
    assert result["development_levy"] == 0
    assert result["total_payable"] == 0


def test_cit_and_levy_worked_example():
    """N500m turnover, N12m assessable profit, N2m capital allowances.

    CIT  = (12,000,000 - 2,000,000) x 30% = 3,000,000
    levy = 12,000,000 x 4% (s. 59: on assessable profits) = 480,000
    """
    result = tax.calculate_cit({"turnover": 500_000_000, "assessable_profit": 12_000_000,
                                "capital_allowances": 2_000_000})

    assert result["taxable_profit"] == 10_000_000
    assert result["tax_payable"] == 3_000_000
    assert result["development_levy"] == 480_000
    assert result["total_payable"] == 3_480_000


def test_fixed_assets_are_required_when_turnover_alone_cannot_decide():
    with pytest.raises(tax.TaxInputError) as excinfo:
        tax.calculate_cit({"turnover": 50_000_000, "assessable_profit": 12_000_000})
    assert "fixed_assets" in str(excinfo.value)


def test_fixed_assets_are_not_needed_above_the_turnover_limit():
    result = tax.calculate_cit({"turnover": 200_000_000, "assessable_profit": 1_000_000})

    assert result["fixed_assets"] is None
    assert result["is_small_company"] is False


def test_allowances_cannot_create_negative_cit():
    result = tax.calculate_cit({"turnover": 500_000_000, "assessable_profit": 1_000_000,
                                "capital_allowances": 9_000_000})

    assert result["tax_payable"] == 0
    assert result["development_levy"] == 40_000  # the levy is on assessable profit


@pytest.mark.parametrize("turnover,flag", [(49_999_999_999, False), (50_000_000_000, True)])
def test_minimum_effective_tax_rate_is_flagged_not_computed(turnover, flag):
    """s. 57: 15% minimum ETR for turnover of N50bn and above, computed from
    audited accounts — so it is flagged rather than calculated."""
    result = tax.calculate_cit({"turnover": turnover, "assessable_profit": 1_000_000})

    assert result["minimum_etr_may_apply"] is flag


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
