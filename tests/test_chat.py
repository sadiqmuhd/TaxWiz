"""Routing tests for the conversational assistant (chat.py, /api/chat).

The property that matters most: a calculation request must reach tax.py and
must never reach Gemini. Every calculation test therefore runs with the AI
entry points replaced by functions that fail the test if they are called, and
with tax.calculate wrapped in a spy that proves the real engine produced the
figure. Informational tests replace the RAG pipeline with a recorder, so no
test here makes a network call.
"""

import pytest

import chat
import rag
import tax


def _forbidden(name):
    def explode(*args, **kwargs):
        raise AssertionError("%s must not be called for a calculation request" % name)
    return explode


@pytest.fixture
def no_ai(monkeypatch):
    """Any attempt to embed, retrieve, generate or ask the RAG pipeline fails."""
    for name in ("answer_question", "embed", "embed_texts", "retrieve", "generate"):
        monkeypatch.setattr(rag, name, _forbidden("rag.%s" % name))


@pytest.fixture
def calc_spy(monkeypatch):
    """Record every call into the real calculation engine, then run it."""
    calls = []
    real = tax.calculate

    def spy(tax_type, payload):
        calls.append((tax_type, dict(payload)))
        return real(tax_type, payload)

    monkeypatch.setattr(tax, "calculate", spy)
    return calls


@pytest.fixture
def rag_recorder(monkeypatch):
    """Stand-in for the RAG pipeline that records the questions it receives."""
    questions = []

    def answer(question):
        questions.append(question)
        return {"answer": "PAYE is a tax deducted from salaries. [1]",
                "sources": [{"id": "chunk-1", "score": 0.7, "excerpt": "..."}],
                "grounded": True}

    monkeypatch.setattr(rag, "answer_question", answer)
    monkeypatch.setattr(rag, "generate", _forbidden("rag.generate"))
    return questions


def converse(*messages):
    """Send messages in turn, carrying `pending` the way the browser does."""
    pending, reply = None, None
    for message in messages:
        reply = chat.respond(message, pending)
        pending = reply.get("pending")
    return reply


# ── Calculation requests reach tax.py, and only tax.py ─────────────────────────

def test_monthly_paye_request_is_calculated_by_the_engine(no_ai, calc_spy):
    reply = chat.respond("I earn ₦500,000 monthly. How much tax should I pay?")

    assert reply["type"] == "calculation"
    assert calc_spy == [("paye", {"gross_income": 6_000_000})]
    # Fourth Schedule, no deductions: 800,000 @ 0% + 2,200,000 @ 15% + 3,000,000 @ 18%
    # = 0 + 330,000 + 540,000 = 870,000.
    assert reply["calculation"]["annual_tax"] == 870_000
    assert reply["calculation"] == tax.calculate_paye({"gross_income": 6_000_000})
    assert "annualised" in reply["reply"]


def test_explicit_paye_request_with_pension(no_ai, calc_spy):
    reply = chat.respond("I earn ₦500,000 monthly and pay ₦40,000 pension. Calculate PAYE")

    assert calc_spy == [("paye", {"gross_income": 6_000_000, "pension": 480_000})]
    # chargeable 5,520,000: 330,000 + 2,520,000 @ 18% (453,600) = 783,600
    assert reply["calculation"]["annual_tax"] == 783_600


def test_rent_is_read_and_passed_to_the_engine(no_ai, calc_spy):
    reply = chat.respond("I earn ₦500,000 monthly and pay ₦1,000,000 rent a year. "
                         "Calculate my PAYE")

    assert calc_spy == [("paye", {"gross_income": 6_000_000, "annual_rent": 1_000_000})]
    # rent relief 200,000 -> chargeable 5,800,000 -> 330,000 + 504,000 = 834,000
    assert reply["calculation"]["annual_tax"] == 834_000
    assert any(row[0].startswith("Rent relief") for row in reply["breakdown"])


def test_monthly_rent_is_annualised_like_income(no_ai, calc_spy):
    chat.respond("Calculate PAYE: salary ₦500,000 monthly, rent ₦100,000 monthly")

    assert calc_spy[0][1]["annual_rent"] == 1_200_000


def test_low_incomes_mention_the_minimum_wage_exemption(no_ai):
    reply = chat.respond("Calculate PAYE on ₦70,000 monthly")

    assert "minimum wage" in reply["reply"]
    assert "s. 58" in reply["reply"]


def test_yearly_income_is_not_multiplied(no_ai, calc_spy):
    chat.respond("Calculate PAYE on a salary of ₦6m per annum")

    assert calc_spy == [("paye", {"gross_income": 6_000_000})]


def test_vat_request(no_ai, calc_spy):
    reply = chat.respond("Calculate VAT on a transaction worth ₦100,000.")

    assert calc_spy == [("vat", {"amount": 100_000, "mode": "exclusive"})]
    assert reply["calculation"]["vat_amount"] == 7_500
    assert "excluding VAT" in reply["reply"]  # the default is stated, not hidden


def test_vat_inclusive_request(no_ai, calc_spy):
    reply = chat.respond("Calculate VAT on ₦537,500 inclusive")

    assert calc_spy == [("vat", {"amount": 537_500, "mode": "inclusive"})]
    assert reply["calculation"]["net_amount"] == 500_000


def test_cit_request(no_ai, calc_spy):
    reply = chat.respond("Calculate CIT: turnover ₦500m, profit ₦12m, capital allowances ₦2m")

    assert calc_spy == [("cit", {"turnover": 500_000_000, "assessable_profit": 12_000_000,
                                  "capital_allowances": 2_000_000})]
    # CIT 10,000,000 x 30% = 3,000,000; levy 12,000,000 x 4% = 480,000
    assert reply["calculation"]["total_payable"] == 3_480_000
    assert "development levy" in reply["reply"]


def test_small_company_needs_fixed_assets_then_pays_nothing(no_ai, calc_spy):
    first = chat.respond("Calculate CIT: turnover ₦50m, profit ₦12m")
    assert first["type"] == "clarification"
    assert "fixed assets" in first["reply"]
    assert calc_spy == []

    reply = chat.respond("₦30m", first["pending"])

    assert calc_spy == [("cit", {"turnover": 50_000_000, "assessable_profit": 12_000_000,
                                  "fixed_assets": 30_000_000})]
    assert reply["calculation"]["is_small_company"] is True
    assert reply["calculation"]["total_payable"] == 0
    assert "small company" in reply["reply"]


def test_wht_request(no_ai, calc_spy):
    reply = chat.respond("Calculate WHT on a ₦1m consultancy fee")

    assert calc_spy == [("wht", {"amount": 1_000_000, "transaction_type": "consultancy"})]
    assert reply["calculation"]["wht_deducted"] == 50_000


@pytest.mark.parametrize("message", [
    "Calculate VAT on ₦100,000",
    "Work out the VAT on ₦100k",
    "What is the VAT on N100,000?",
    "How much VAT is charged on 100,000 naira?",
    "Compute VAT for ₦0.1m",
])
def test_natural_variations_of_a_vat_request(no_ai, calc_spy, message):
    reply = chat.respond(message)

    assert reply["type"] == "calculation"
    assert calc_spy[0][1]["amount"] == 100_000


def test_every_calculation_carries_the_estimate_notice(no_ai):
    reply = chat.respond("Calculate VAT on ₦100,000")

    assert reply["notice"] == chat.ESTIMATE_NOTICE
    assert "not been independently verified" in reply["notice"]


def test_a_rate_in_the_message_is_not_used(no_ai, calc_spy):
    """The user cannot set the rate; the built-in one applies, and we say so."""
    reply = chat.respond("Calculate VAT at 10% on ₦100,000")

    assert reply["calculation"]["vat_amount"] == 7_500
    assert "built-in rates" in reply["reply"]


# ── Informational questions go to RAG ──────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "What is PAYE?",
    "Explain how VAT works in Nigeria.",
    "Who is liable to pay company income tax?",
    "What are the differences between PAYE and withholding tax?",
    "What is the VAT rate?",
    "How much is VAT in Nigeria?",
    "Is VAT charged on a sale of ₦100,000?",
    "What does the Tax Act 2025 say about VAT?",
    "What is tax on dividends?",
])
def test_informational_questions_use_rag(rag_recorder, calc_spy, question):
    reply = chat.respond(question)

    assert reply["type"] == "answer"
    assert rag_recorder == [question]
    assert calc_spy == []


def test_rag_sources_and_grounding_pass_through_unchanged(rag_recorder):
    reply = chat.respond("What is PAYE?")

    assert reply["grounded"] is True
    assert reply["sources"][0]["id"] == "chunk-1"
    assert reply["pending"] is None


def test_an_ungrounded_rag_result_stays_ungrounded(monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda q: {
        "answer": rag.NO_ANSWER, "sources": [], "grounded": False})

    reply = chat.respond("What is the capital of France?")

    assert reply["grounded"] is False
    assert reply["sources"] == []


# ── Questions the corpus cannot answer ─────────────────────────────────────────

@pytest.fixture
def empty_corpus(monkeypatch):
    """RAG finds nothing it can cite — the real situation for "What is VAT?",
    because the indexed corpus is only the Act's definitions section."""
    questions = []

    def answer(question):
        questions.append(question)
        return {"answer": rag.NO_ANSWER, "sources": [], "grounded": False}

    monkeypatch.setattr(rag, "answer_question", answer)
    return questions


@pytest.mark.parametrize("question", ["What is VAT?", "VAT", "Explain value added tax"])
def test_an_unsourced_tax_question_offers_the_calculator(empty_corpus, calc_spy, question):
    reply = chat.respond(question)

    assert empty_corpus == [question]          # RAG was still consulted first
    assert reply["type"] == "answer"
    assert reply["grounded"] is False
    assert reply["sources"] == []
    assert "won't answer from memory" in reply["reply"]
    assert "Calculate VAT on" in reply["reply"]
    assert "not been independently verified" in reply["reply"]
    assert calc_spy == []                      # an offer, not a calculation


def test_the_offer_quotes_the_engines_rate_rather_than_a_copy(empty_corpus, monkeypatch):
    """If tax.py's rate changes, the chat must say the new rate."""
    monkeypatch.setattr(tax, "VAT_RATE", 0.1)

    assert "VAT at 10%" in chat.respond("What is VAT?")["reply"]


def test_the_offer_covers_every_tax_named(empty_corpus):
    reply = chat.respond("What are the differences between PAYE and withholding tax?")["reply"]

    assert "PAYE" in reply and "withholding tax" in reply
    assert "0% to 25%" in reply and "5% or 10%" in reply


def test_an_unsourced_question_naming_no_tax_gets_the_plain_refusal(empty_corpus):
    reply = chat.respond("What is the capital of France?")

    assert reply["reply"] == rag.NO_ANSWER


def test_a_grounded_answer_is_never_replaced_by_the_offer(rag_recorder):
    reply = chat.respond("What is PAYE?")

    assert reply["reply"] == "PAYE is a tax deducted from salaries. [1]"


# ── Missing information is asked for, never invented ───────────────────────────

def test_paye_without_income_asks_for_it(no_ai, calc_spy):
    reply = chat.respond("Calculate my PAYE tax.")

    assert reply["type"] == "clarification"
    assert "gross income" in reply["reply"]
    assert calc_spy == []


def test_paye_without_a_period_asks_monthly_or_yearly(no_ai, calc_spy):
    """₦500,000 a month and ₦500,000 a year differ twelve-fold. Never guess."""
    reply = chat.respond("Calculate PAYE on ₦500,000")

    assert reply["type"] == "clarification"
    assert "monthly or your yearly" in reply["reply"]
    assert calc_spy == []


def test_cit_without_figures_asks_for_them(no_ai, calc_spy):
    reply = chat.respond("What is the company income tax on my business profit?")

    assert reply["type"] == "clarification"
    assert "turnover" in reply["reply"] and "assessable profit" in reply["reply"]
    assert calc_spy == []


def test_wht_without_amount_or_type_asks_for_both(no_ai, calc_spy):
    reply = chat.respond("Calculate withholding tax on this payment.")

    assert reply["type"] == "clarification"
    assert "payment amount" in reply["reply"] and "Consultancy" in reply["reply"]
    assert calc_spy == []


def test_vat_without_amount_asks_for_it(no_ai, calc_spy):
    reply = chat.respond("Calculate VAT for me")

    assert reply["type"] == "clarification"
    assert calc_spy == []


def test_a_year_is_not_read_as_money(no_ai, calc_spy):
    reply = chat.respond("Calculate VAT under the Tax Act 2025")

    assert reply["type"] == "clarification"
    assert calc_spy == []


def test_net_pay_is_not_used_as_gross(no_ai, calc_spy):
    """Treating take-home pay as gross would understate the income."""
    reply = chat.respond("Calculate PAYE, my take-home is ₦400k monthly")

    assert reply["type"] == "clarification"
    assert "gross" in reply["reply"]
    assert calc_spy == []


# ── Multi-turn completion ──────────────────────────────────────────────────────

def test_paye_completed_over_three_turns(no_ai, calc_spy):
    reply = converse("Calculate my PAYE", "₦500,000", "monthly")

    assert reply["type"] == "calculation"
    assert calc_spy == [("paye", {"gross_income": 6_000_000})]
    assert reply["pending"] is None


def test_cit_completed_after_a_clarification(no_ai, calc_spy):
    reply = converse("What is the company income tax on my business profit?",
                     "turnover ₦50m, profit ₦12m",
                     "fixed assets of ₦400m")

    assert calc_spy == [("cit", {"turnover": 50_000_000, "assessable_profit": 12_000_000,
                                  "fixed_assets": 400_000_000})]
    # Turnover is within the small-company limit but assets are not, so the
    # company is taxed: 12,000,000 x 30% = 3,600,000; levy x 4% = 480,000.
    assert reply["calculation"]["is_small_company"] is False
    assert reply["calculation"]["total_payable"] == 4_080_000


def test_a_single_unlabelled_figure_fills_the_one_missing_slot(no_ai, calc_spy):
    # Above N100m turnover no fixed-assets figure is needed, so profit is the
    # only missing slot and a bare amount can fill it.
    reply = converse("Calculate CIT, my turnover is ₦500m", "₦12m")

    assert calc_spy == [("cit", {"turnover": 500_000_000, "assessable_profit": 12_000_000})]
    assert reply["type"] == "calculation"


def test_wht_completed_over_turns(no_ai, calc_spy):
    reply = converse("Calculate withholding tax on this payment.", "₦1,000,000", "it's rent")

    assert calc_spy == [("wht", {"amount": 1_000_000, "transaction_type": "dividends"})]
    assert reply["calculation"]["wht_deducted"] == 100_000


def test_choosing_the_tax_reuses_the_original_figures(no_ai, calc_spy):
    """'Income tax' could be personal or company tax: ask, then use the answer."""
    first = chat.respond("Calculate my income tax on ₦6m a year")
    assert first["type"] == "clarification"
    assert set(first["pending"]["choose_from"]) == {"paye", "cit"}

    reply = chat.respond("PAYE", first["pending"])

    assert calc_spy == [("paye", {"gross_income": 6_000_000})]
    assert reply["type"] == "calculation"


def test_an_informational_question_mid_calculation_goes_to_rag(rag_recorder, calc_spy):
    first = chat.respond("Calculate my PAYE")
    reply = chat.respond("What is PAYE?", first["pending"])

    assert reply["type"] == "answer"
    assert rag_recorder == ["What is PAYE?"]
    assert calc_spy == []


def test_a_new_explicit_request_replaces_the_pending_one(no_ai, calc_spy):
    first = chat.respond("Calculate my PAYE")
    reply = chat.respond("Calculate VAT on ₦100,000", first["pending"])

    assert calc_spy == [("vat", {"amount": 100_000, "mode": "exclusive"})]
    assert reply["type"] == "calculation"


def test_cancel_clears_the_pending_calculation(no_ai, calc_spy):
    first = chat.respond("Calculate my PAYE")
    reply = chat.respond("cancel", first["pending"])

    assert reply["pending"] is None
    assert calc_spy == []


# ── Invalid, ambiguous and unsupported requests are handled safely ─────────────

def test_negative_amount_is_rejected_by_the_engine_and_explained(no_ai):
    reply = chat.respond("Calculate PAYE on ₦-500,000 monthly")

    assert reply["type"] == "clarification"
    assert "cannot be negative" in reply["reply"]
    assert "calculation" not in reply


def test_zero_amount_is_rejected(no_ai):
    reply = chat.respond("Calculate VAT on ₦0")

    assert reply["type"] == "clarification"
    assert "greater than zero" in reply["reply"]


def test_two_taxes_in_one_request_asks_which(no_ai, calc_spy):
    reply = chat.respond("Calculate PAYE and VAT on ₦500k")

    assert reply["type"] == "clarification"
    assert set(reply["pending"]["choose_from"]) == {"paye", "vat"}
    assert calc_spy == []


def test_a_tax_request_naming_no_tax_asks_which(no_ai, calc_spy):
    reply = chat.respond("How much tax will I owe on ₦250,000?")

    assert reply["type"] == "clarification"
    assert set(reply["pending"]["choose_from"]) == set(tax.CALCULATORS)
    assert calc_spy == []


def test_several_amounts_for_vat_asks_which(no_ai, calc_spy):
    reply = chat.respond("Calculate VAT on ₦100,000 and ₦200,000")

    assert reply["type"] == "clarification"
    assert calc_spy == []


def test_an_ambiguous_wht_category_asks_which(no_ai, calc_spy):
    reply = chat.respond("Calculate WHT on ₦1m of rent and director fees")

    assert reply["type"] == "clarification"
    assert calc_spy == []


def test_a_bare_amount_asks_what_to_do_with_it(no_ai, calc_spy):
    reply = chat.respond("₦500,000")

    assert reply["type"] == "clarification"
    assert reply["pending"] is None
    assert calc_spy == []


def test_monthly_cit_figures_are_questioned(no_ai, calc_spy):
    reply = chat.respond("Calculate CIT: monthly turnover ₦5m, profit ₦1m")

    assert reply["type"] == "clarification"
    assert "annual" in reply["reply"]
    assert calc_spy == []


@pytest.mark.parametrize("message", [
    "Calculate capital gains tax on ₦5m",
    "Calculate stamp duty on my lease",
    "Calculate education tax for my company",
])
def test_unsupported_taxes_are_declined(no_ai, calc_spy, message):
    reply = chat.respond(message)

    assert reply["type"] == "clarification"
    assert "can't calculate" in reply["reply"]
    assert calc_spy == []


@pytest.mark.parametrize("message", ["", "   "])
def test_blank_messages_are_rejected(message):
    with pytest.raises(ValueError):
        chat.respond(message)


def test_overlong_messages_are_rejected():
    with pytest.raises(ValueError):
        chat.respond("calculate " * 200)


# ── The client-supplied pending object is untrusted ────────────────────────────

@pytest.mark.parametrize("pending", [
    "not a dict",
    {"tax_type": "lottery", "slots": {}},
    {"tax_type": "vat", "slots": {"amount": "100000"}},
    {"tax_type": "vat", "slots": {"amount": float("nan")}},
    {"tax_type": "vat", "slots": {"amount": True}},
    {"tax_type": "paye", "slots": {"gross_period": "fortnightly"}},
    {"tax_type": "wht", "slots": {"transaction_type": "bribes"}},
    {"choose_from": ["paye", "lottery"], "message": "x"},
    {"choose_from": [], "message": "x"},
])
def test_malformed_pending_state_is_discarded(pending):
    assert chat.validate_pending(pending) is None


def test_unknown_slot_keys_are_dropped():
    state = chat.validate_pending({"tax_type": "vat",
                                   "slots": {"amount": 100, "rate": 0.5}})

    assert state == {"tax_type": "vat", "slots": {"amount": 100.0}}


def test_a_tampered_pending_object_cannot_bypass_validation(no_ai):
    """Even a hand-crafted pending object goes through tax.py's validation."""
    reply = chat.respond("monthly", {"tax_type": "paye",
                                     "slots": {"gross_amount": -1_000_000}})

    assert reply["type"] == "clarification"
    assert "cannot be negative" in reply["reply"]


# ── Reading amounts ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("₦500,000", [500_000]),
    ("N500,000", [500_000]),
    ("500k", [500_000]),
    ("1.5m", [1_500_000]),
    ("₦2 million", [2_000_000]),
    ("1.2bn", [1_200_000_000]),
    ("250,000 naira", [250_000]),
    ("₦1,234.50", [1_234.5]),
    ("Tax Act 2025", []),
    ("12 months", []),
    ("7.5%", []),
    ("section 25", []),
    ("I have 2 jobs", []),
    ("₦100,000-₦200,000", [100_000, 200_000]),
])
def test_amount_reading(text, expected):
    amounts, _ = chat.find_amounts(text)
    assert [a["value"] for a in amounts] == expected


def test_percentages_are_noticed_but_not_read_as_money():
    amounts, saw_percent = chat.find_amounts("VAT at 7.5% on ₦1,000")

    assert [a["value"] for a in amounts] == [1_000]
    assert saw_percent is True


# ── The HTTP endpoint ──────────────────────────────────────────────────────────

def test_chat_endpoint_returns_a_calculation(client, no_ai):
    response = client.post("/api/chat", json={"message": "Calculate VAT on ₦100,000"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["type"] == "calculation"
    assert body["calculation"]["vat_amount"] == 7_500
    assert ["VAT (7.5%)", "₦7,500.00"] in body["breakdown"]


def test_chat_endpoint_carries_pending_across_requests(client, no_ai):
    first = client.post("/api/chat", json={"message": "Calculate my PAYE"}).get_json()
    second = client.post("/api/chat", json={"message": "₦6,000,000 a year",
                                            "pending": first["pending"]}).get_json()

    assert first["type"] == "clarification"
    assert second["type"] == "calculation"
    assert second["calculation"]["gross_income"] == 6_000_000


def test_chat_endpoint_routes_questions_to_rag(client, rag_recorder):
    body = client.post("/api/chat", json={"message": "What is PAYE?"}).get_json()

    assert body["type"] == "answer"
    assert rag_recorder == ["What is PAYE?"]


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": 42}])
def test_chat_endpoint_rejects_a_missing_message(client, payload):
    response = client.post("/api/chat", json=payload)

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid request"


def test_chat_endpoint_reports_rag_outages_cleanly(client, monkeypatch):
    def outage(question):
        raise rag.RagError("The language model is unavailable right now.")

    monkeypatch.setattr(rag, "answer_question", outage)
    response = client.post("/api/chat", json={"message": "What is PAYE?"})

    assert response.status_code == 502
    assert "unavailable" in response.get_json()["message"]


def test_calculations_still_work_when_the_ai_is_down(client, monkeypatch):
    """The calculator does not depend on Gemini or Pinecone at all."""
    def outage(*args, **kwargs):
        raise rag.RagError("down")

    monkeypatch.setattr(rag, "answer_question", outage)
    response = client.post("/api/chat", json={"message": "Calculate VAT on ₦100,000"})

    assert response.status_code == 200
    assert response.get_json()["type"] == "calculation"
