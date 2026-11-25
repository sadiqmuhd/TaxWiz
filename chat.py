"""The conversational front door: one chat message in, one reply out.

Every message is routed to exactly one of two engines:

  * a calculation request  -> figures read from the text -> tax.calculate()
  * an informational question -> rag.answer_question() (Gemini + Pinecone)

Routing and figure extraction are plain rules — regular expressions and keyword
lists. No language model decides whether to calculate, reads the numbers, or
produces a figure: a model cannot be talked into a different tax result, and
every routing decision can be tested and explained. Gemini is used only for
informational questions, through the unchanged RAG pipeline.

When a calculation is missing information, the reply asks for it and returns a
small `pending` object. The client sends that object back with its next message,
so the server stays stateless (gunicorn runs several workers, and in-memory
conversation state would not survive from one request to the next). `pending`
comes from the client, so it is treated as untrusted and re-validated.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import rag
import tax

ESTIMATE_NOTICE = (
    "Estimate only. TaxWiz's rules are its own reading of the Nigeria Tax Act 2025 "
    "and have not been independently verified by a tax professional. Confirm your "
    "actual liability with the Nigeria Revenue Service (NRS) or a qualified tax "
    "professional."
)

TAX_NAMES = {
    "paye": "PAYE (personal income tax)",
    "cit": "company income tax",
    "vat": "VAT",
    "wht": "withholding tax",
}

# Figures each calculator cannot run without. PAYE needs to know whether the
# income figure is monthly or yearly: guessing wrong is a twelve-fold error.
# CIT additionally needs fixed assets when turnover alone cannot decide whether
# the company is small (see _missing).
REQUIRED_SLOTS = {
    "paye": ("gross_amount", "gross_period"),
    "cit": ("turnover", "assessable_profit"),
    "vat": ("amount",),
    "wht": ("amount", "transaction_type"),
}
NUMERIC_SLOTS = {
    "paye": ("gross_amount", "pension_amount", "nhf_amount", "nhis_amount", "rent_amount"),
    "cit": ("turnover", "assessable_profit", "capital_allowances", "fixed_assets"),
    "vat": ("amount",),
    "wht": ("amount",),
}
PERIOD_SLOTS = ("gross_period", "pension_period", "nhf_period", "nhis_period", "rent_period")
PERIODS = ("monthly", "annual")

# PAYE deductions the chat understands: slot prefix -> (tax.py field, wording).
# Mortgage interest and life assurance are accepted by tax.py and the API, but
# not read from chat messages.
CHAT_DEDUCTIONS = {
    "pension": ("pension", "pension contribution"),
    "nhf": ("nhf", "NHF contribution"),
    "nhis": ("nhis", "NHIS contribution"),
    "rent": ("annual_rent", "rent"),
}

# The minimum-wage exemption (s. 58) cannot be applied — the Act does not state
# the wage — so below this income the reply mentions it. Display only.
MIN_WAGE_NOTE_BELOW = 1_500_000


# ── Patterns ───────────────────────────────────────────────────────────────────

FLAGS = re.IGNORECASE

# Tax names, matched longest-first and masked out of the text once found, so
# "company income tax" is not also read as "income tax", and the word "income"
# inside a tax name is not mistaken for a description of an amount.
CIT_NAME = re.compile(r"\bcompany(?:'s)? income tax\b|\bcorporate (?:income )?tax\b|"
                      r"\bcompany(?:'s)? tax\b|\bcit\b", FLAGS)
PAYE_NAME = re.compile(r"\bpersonal income tax\b|\bpay[- ]as[- ]you[- ]earn\b|\bpaye\b", FLAGS)
GENERIC_INCOME_TAX = re.compile(r"\bincome tax\b", FLAGS)
VAT_NAME = re.compile(r"\bvalue[- ]added tax\b|\bvat\b", FLAGS)
WHT_NAME = re.compile(r"\bwithholding(?: tax)?\b|\bwht\b", FLAGS)

UNSUPPORTED = re.compile(
    r"\b(capital gains tax|cgt|stamp dut(?:y|ies)|tertiary education tax|education tax|"
    r"customs dut(?:y|ies)|import dut(?:y|ies)|excise dut(?:y|ies)|excise|"
    r"petroleum profits? tax|land use charge|property tax|inheritance tax|"
    r"development levy)\b", FLAGS)

CALC_VERB = re.compile(r"\b(calculat\w*|comput\w*|work(?:ing)? out|estimat\w*|figure out)\b", FLAGS)
HOW_MUCH = re.compile(r"\bhow much\b", FLAGS)
OWE = re.compile(r"\b(pay|paying|owe|due|deduct\w*|charg\w*|withh[eo]ld\w*|remit\w*)\b", FLAGS)
# "VAT on a transaction", "tax on my profit", "WHT on ₦1m": asking for a figure.
TAX_ON = re.compile(r"\b(?:tax|vat|paye|wht|cit)\b[^.?!]{0,15}?\b(?:on|for|of)\s+"
                    r"(?:(?:my|our|this|that|these|a|an)\b|(?=[₦\d])|(?=n\d)|(?=ngn))", FLAGS)
# Openers that make a message a question about the law, not a request for a figure.
QUESTION_START = re.compile(
    r"^\s*(is|are|was|were|does|do|did|can|could|should|would|will|why|who|whom|when|"
    r"where|which|define|explain|describe|tell me about|what does|"
    r"what (?:is|are) (?:the )?differences?)\b", FLAGS)
CANCEL = re.compile(r"^\s*(cancel|never ?mind|stop|forget it|start over|reset)\b", FLAGS)

PAYE_HINT = re.compile(r"\b(earn\w*|salar\w*|wages?|take[- ]home|employ\w*|personal)\b", FLAGS)
# PAYE is computed from gross (before-tax) pay. A net figure cannot be used as
# gross without understating the income, so it triggers a question instead.
NET_PAY = re.compile(r"\btake[- ]home\b|\bnet (?:pay|salary|income|earnings)\b|\bafter tax\b", FLAGS)
CIT_HINT = re.compile(r"\b(business\w*|compan(?:y|ies)|firm|turnover|revenue|profits?)\b", FLAGS)

AMOUNT = re.compile(
    r"(?<![\w.,])"
    r"(?P<sign>-\s?)?"
    r"(?P<cur>₦\s?|ngn\s?|n(?=\d))?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<suffix>k|thousand|m|mn|million|b|bn|billion)\b)?"
    r"(?P<pct>\s?(?:%|percent\b|per cent\b))?", FLAGS)
MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6,
               "b": 1e9, "bn": 1e9, "billion": 1e9}
NAIRA_AFTER = re.compile(r"^\s?(naira|ngn)\b", FLAGS)
NOT_MONEY_AFTER = re.compile(r"^\s?(months?|years?|yrs?|weeks?|days?|th|st|nd|rd)\b", FLAGS)
NOT_MONEY_BEFORE = re.compile(r"\b(section|s\.|schedule|paragraph|chapter|part|article|"
                              r"act|item|no\.?|number)\s*$", FLAGS)

MONTHLY = re.compile(r"\b(monthly|per month|a month|each month|every month|per mth)\b|"
                     r"/\s*(?:month|mo)\b", FLAGS)
ANNUAL = re.compile(r"\b(annual(?:ly)?|yearly|per year|a year|each year|every year|"
                    r"per annum|this year|last year)\b|\bp\.a\.|/\s*(?:year|yr)\b", FLAGS)

PAYE_LABELS = {
    "pension": re.compile(r"\bpension\w*|\brsa\b|\bretirement savings?\b", FLAGS),
    "nhf": re.compile(r"\bnhf\b|\bhousing fund\b|\bnational housing\b", FLAGS),
    "nhis": re.compile(r"\bnhis\b|\bhealth insurance\b", FLAGS),
    "rent": re.compile(r"\brent(?:s|al)?\b", FLAGS),
    "gross": re.compile(r"\b(earn\w*|salar\w*|wages?|income|gross)\b", FLAGS),
}
CIT_LABELS = {
    "turnover": re.compile(r"\b(turnover|revenue|sales|gross receipts)\b", FLAGS),
    "assessable_profit": re.compile(r"\b(assessable profits?|profits?)\b", FLAGS),
    "capital_allowances": re.compile(r"\bcapital allowances?\b|\ballowances?\b", FLAGS),
    "fixed_assets": re.compile(r"\b(?:fixed|total) assets?\b|\bassets?\b", FLAGS),
}
VAT_INCLUSIVE = re.compile(r"\b(inclusive|including vat|includes vat|incl\.? vat|"
                           r"vat[- ]inclusive|vat included|already includes)\b", FLAGS)
VAT_EXCLUSIVE = re.compile(r"\b(exclusive|excluding vat|excludes vat|excl\.? vat|before vat|"
                           r"plus vat|add vat|on top|without vat|vat[- ]exclusive)\b", FLAGS)
WHT_TYPES = {
    "dividends": re.compile(r"\b(dividends?|interest|rent|rental|lease)\b", FLAGS),
    "director": re.compile(r"\bdirectors?\b", FLAGS),
    "commission": re.compile(r"\b(commissions?|royalt(?:y|ies))\b", FLAGS),
    "construction": re.compile(r"\b(construction|survey\w*)\b", FLAGS),
    "consultancy": re.compile(r"\b(consult\w*|management fees?|management services?)\b", FLAGS),
    "professional": re.compile(r"\b(professional|technical|legal fees?|audit\w*)\b", FLAGS),
    "contracts": re.compile(r"\b(contracts?|suppl(?:y|ies)|goods)\b", FLAGS),
}


# ── Reading the message ────────────────────────────────────────────────────────

def _mask(text: str, pattern) -> Tuple[str, int]:
    """Blank out every match (keeping offsets stable) and count them."""
    count = 0

    def blank(match):
        nonlocal count
        count += 1
        return " " * len(match.group(0))

    return pattern.sub(blank, text), count


def find_tax_types(text: str) -> Tuple[set, bool, str]:
    """Supported tax types named in `text`.

    Returns (types, generic_income_tax, masked_text). A bare "income tax" could
    be personal or company tax, so it is reported separately for the caller to
    resolve rather than guessed.
    """
    types = set()
    masked = text
    for key, pattern in (("cit", CIT_NAME), ("paye", PAYE_NAME)):
        masked, hits = _mask(masked, pattern)
        if hits:
            types.add(key)
    masked, generic = _mask(masked, GENERIC_INCOME_TAX)
    for key, pattern in (("vat", VAT_NAME), ("wht", WHT_NAME)):
        masked, hits = _mask(masked, pattern)
        if hits:
            types.add(key)
    return types, bool(generic), masked


def find_amounts(text: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Money amounts in `text`, and whether a percentage was mentioned.

    Deliberately conservative: years ("Act 2025"), counts ("12 months"),
    section numbers and small bare numbers are not treated as money. A missed
    amount produces a clarification question; a wrong one produces a wrong tax.
    """
    amounts, saw_percent = [], False
    for match in AMOUNT.finditer(text):
        if match.group("pct"):
            saw_percent = True
            continue
        after, before = text[match.end():], text[:match.start()]
        if NOT_MONEY_AFTER.match(after) or NOT_MONEY_BEFORE.search(before):
            continue
        number = match.group("num")
        marked = bool(match.group("cur") or match.group("suffix") or NAIRA_AFTER.match(after))
        value = float(number.replace(",", ""))
        if not marked and "," not in number:
            if value < 100 or (value.is_integer() and 1900 <= value <= 2099):
                continue
        suffix = (match.group("suffix") or "").lower()
        value *= MULTIPLIERS.get(suffix, 1)
        if match.group("sign"):
            value = -value
        amounts.append({"value": value, "start": match.start(), "end": match.end()})
    return amounts, saw_percent


def _distance(span: Tuple[int, int], amount: Dict[str, Any]) -> int:
    start, end = span
    if end <= amount["start"]:
        return amount["start"] - end
    if amount["end"] <= start:
        return start - amount["end"]
    return 0


def _attach(text: str, patterns: Dict[str, Any], amounts) -> Dict[int, str]:
    """Pair each keyword with the amount nearest to it.

    "I earn ₦500,000 and pay ₦40,000 pension" -> "earn" belongs to ₦500,000
    and "pension" to ₦40,000. An amount near several keywords takes the closest.
    """
    best: Dict[int, Tuple[int, str]] = {}
    if not amounts:
        return {}
    for label, pattern in patterns.items():
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            nearest = min(range(len(amounts)), key=lambda i: _distance(span, amounts[i]))
            distance = _distance(span, amounts[nearest])
            if nearest not in best or distance < best[nearest][0]:
                best[nearest] = (distance, label)
    return {index: label for index, (_, label) in best.items()}


def _period_mentions(text: str) -> set:
    kinds = set()
    if MONTHLY.search(text):
        kinds.add("monthly")
    if ANNUAL.search(text):
        kinds.add("annual")
    return kinds


# ── Extracting the figures each calculator needs ───────────────────────────────

def extract_slots(tax_type: str, text: str, masked: str, amounts,
                  known: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Read the inputs for `tax_type` out of a message.

    `known` holds slots already collected earlier in the conversation, so a
    lone amount can be matched to the one figure still missing. Returns
    (new_slots, problems); a problem is a question to put back to the user
    instead of guessing.
    """
    slots: Dict[str, Any] = {}
    problems: List[str] = []

    def missing_amount_slots():
        return [s for s in _missing(tax_type, known) if s in NUMERIC_SLOTS[tax_type]]

    if tax_type == "paye":
        if NET_PAY.search(masked) and amounts:
            problems.append("PAYE is worked out from gross pay, before tax and deductions. "
                            "What is your gross income, and is it monthly or yearly?")
            return slots, problems
        labels = _attach(masked, PAYE_LABELS, amounts)
        periods = _attach(masked, {"monthly": MONTHLY, "annual": ANNUAL}, amounts)
        gross = [i for i in range(len(amounts)) if labels.get(i) in (None, "gross")]
        for field, (_, name) in CHAT_DEDUCTIONS.items():
            hits = [i for i, label in labels.items() if label == field]
            if len(hits) > 1:
                problems.append("I found more than one %s figure. Which one should I use?"
                                % name)
            elif hits:
                slots["%s_amount" % field] = amounts[hits[0]]["value"]
                if hits[0] in periods:
                    slots["%s_period" % field] = periods[hits[0]]
        if len(gross) > 1:
            problems.append("I found several amounts: %s. Which one is your gross income?"
                            % ", ".join(naira(amounts[i]["value"]) for i in gross))
        elif gross:
            slots["gross_amount"] = amounts[gross[0]]["value"]
            if gross[0] in periods:
                slots["gross_period"] = periods[gross[0]]
        # "monthly" answering "is that monthly or yearly?", with no amount given.
        mentioned = _period_mentions(masked)
        if "gross_period" not in slots and not amounts and len(mentioned) == 1:
            slots["gross_period"] = mentioned.pop()
        elif ("gross_amount" in slots and "gross_period" not in slots
              and len(mentioned) == 1 and len(amounts) == 1):
            slots["gross_period"] = mentioned.pop()

    elif tax_type == "cit":
        labels = _attach(masked, CIT_LABELS, amounts)
        unlabelled = [i for i in range(len(amounts)) if i not in labels]
        for field in CIT_LABELS:
            hits = [i for i, label in labels.items() if label == field]
            if len(hits) > 1:
                problems.append("I found more than one figure for %s. Which is right?"
                                % field.replace("_", " "))
            elif hits:
                slots[field] = amounts[hits[0]]["value"]
        still_missing = [s for s in missing_amount_slots() if s not in slots]
        if len(unlabelled) == 1 and len(still_missing) == 1:
            slots[still_missing[0]] = amounts[unlabelled[0]]["value"]
        elif unlabelled:
            problems.append("I'm not sure what %s refers to. Please label each figure, e.g. "
                            "\"turnover ₦50m, profit ₦12m\"."
                            % ", ".join(naira(amounts[i]["value"]) for i in unlabelled))
        if MONTHLY.search(masked):
            problems.append("Company income tax uses annual figures. Please give your "
                            "yearly turnover and profit.")

    else:  # vat, wht: a single transaction amount
        if len(amounts) > 1:
            problems.append("I found several amounts: %s. Which one is the %s?"
                            % (", ".join(naira(a["value"]) for a in amounts),
                               "transaction amount" if tax_type == "vat" else "payment amount"))
        elif amounts:
            slots["amount"] = amounts[0]["value"]

        if tax_type == "vat":
            inclusive, exclusive = VAT_INCLUSIVE.search(masked), VAT_EXCLUSIVE.search(masked)
            if inclusive and exclusive:
                problems.append("Does the amount already include VAT, or should VAT be added on top?")
            elif inclusive:
                slots["mode"] = "inclusive"
            elif exclusive:
                slots["mode"] = "exclusive"
        else:
            kinds = {key for key, pattern in WHT_TYPES.items() if pattern.search(masked)}
            if len(kinds) > 1 and "contracts" in kinds:
                kinds.discard("contracts")  # "consultancy contract" is consultancy
            if len(kinds) > 1:
                problems.append("That could be more than one kind of payment (%s). Which is it?"
                                % ", ".join(tax.WHT_RATES[k]["label"] for k in sorted(kinds)))
            elif kinds:
                slots["transaction_type"] = kinds.pop()

    return slots, problems


def _missing(tax_type: str, slots: Dict[str, Any]) -> List[str]:
    missing = [s for s in REQUIRED_SLOTS[tax_type] if s not in slots]
    # A company with turnover of N100m or less is small only if its fixed assets
    # are also within the limit, so that figure becomes necessary.
    if (tax_type == "cit" and "fixed_assets" not in slots
            and slots.get("turnover", float("inf")) <= tax.SMALL_COMPANY_TURNOVER):
        missing.append("fixed_assets")
    return missing


def _ask_for(tax_type: str, slots: Dict[str, Any]) -> str:
    """The clarification question for whatever is still missing."""
    missing = _missing(tax_type, slots)
    if tax_type == "paye":
        if "gross_amount" in missing:
            return ("To estimate your PAYE I need your gross income, and whether it's monthly "
                    "or yearly — for example \"₦500,000 monthly\" or \"₦6,000,000 a year\". "
                    "You can include pension, NHF, NHIS and rent paid too.")
        return "Is %s your monthly or your yearly income?" % naira(slots["gross_amount"])
    if tax_type == "cit":
        words = {"turnover": "annual turnover", "assessable_profit": "assessable profit",
                 "fixed_assets": "total fixed assets"}
        wanted = [words[s] for s in missing]
        wanted_text = (wanted[0] if len(wanted) == 1
                       else ", ".join(wanted[:-1]) + " and " + wanted[-1])
        if missing == ["fixed_assets"]:
            return ("With turnover of %s or less, a company counts as small only if its fixed "
                    "assets are %s or less. What are its total fixed assets?"
                    % (naira(tax.SMALL_COMPANY_TURNOVER), naira(tax.SMALL_COMPANY_FIXED_ASSETS)))
        return ("To estimate company income tax I need your %s (capital allowances are "
                "optional) — for example \"turnover ₦50m, fixed assets ₦30m, profit ₦12m\"."
                % wanted_text)
    if tax_type == "vat":
        return ("What is the transaction amount? Tell me too if it already includes VAT — "
                "otherwise I'll add VAT on top.")
    options = "; ".join("%s (%g%%)" % (v["label"], v["rate"] * 100)
                        for v in tax.WHT_RATES.values())
    if missing == ["transaction_type"]:
        return "What kind of payment is it? I can handle: %s." % options
    if missing == ["amount"]:
        return "What is the payment amount?"
    return "What is the payment amount, and what kind of payment is it? I can handle: %s." % options


# ── Running the calculation (through tax.py only) ──────────────────────────────

def _annual(amount: float, period: str) -> float:
    """Input normalisation, not a tax rule: the calculator takes yearly figures."""
    return amount * 12 if period == "monthly" else amount


def calculate(tax_type: str, slots: Dict[str, Any], saw_percent: bool) -> Dict[str, Any]:
    """Build the calculator payload from the collected slots and call tax.py.

    Raises tax.TaxInputError when the figures fail tax.py's own validation.
    """
    notes: List[str] = []
    if tax_type == "paye":
        period = slots["gross_period"]
        payload = {"gross_income": _annual(slots["gross_amount"], period)}
        if period == "monthly":
            notes.append("I annualised your income: %s × 12 = %s."
                         % (naira(slots["gross_amount"]), naira(payload["gross_income"])))
        for field, (key, name) in CHAT_DEDUCTIONS.items():
            if "%s_amount" % field in slots:
                own = slots.get("%s_period" % field)
                payload[key] = _annual(slots["%s_amount" % field], own or period)
                if not own and period == "monthly":
                    notes.append("I treated your %s as monthly, like your income." % name)
    elif tax_type == "cit":
        payload = {k: slots[k] for k in NUMERIC_SLOTS["cit"] if k in slots}
    elif tax_type == "vat":
        payload = {"amount": slots["amount"], "mode": slots.get("mode", "exclusive")}
    else:
        payload = {"amount": slots["amount"], "transaction_type": slots["transaction_type"]}

    result = tax.calculate(tax_type, payload)

    if saw_percent:
        notes.append("I always use TaxWiz's built-in rates; a rate in your message was not applied.")
    reply, breakdown = describe(tax_type, result, notes)
    return {"type": "calculation", "tax_type": tax_type, "reply": reply,
            "breakdown": breakdown, "notice": ESTIMATE_NOTICE,
            "calculation": result, "pending": None}


def describe(tax_type: str, r: Dict[str, Any], notes: List[str]) -> Tuple[str, List[List[str]]]:
    """Turn a tax.py result into words and table rows. No arithmetic here."""
    if tax_type == "paye":
        if r["gross_income"] <= MIN_WAGE_NOTE_BELOW:
            notes.append("If you earn the national minimum wage or less, your employment "
                         "income is exempt (Nigeria Tax Act 2025, s. 58). I haven't applied "
                         "that, because the Act doesn't state the wage amount.")
        reply = ("Estimated PAYE: %s a year, about %s a month (effective rate %s%%)."
                 % (naira(r["annual_tax"]), naira(r["monthly_tax"]), r["effective_rate"]))
        rows = [["Gross annual income", naira(r["gross_income"])]]
        for key, label in tax.DEDUCTIONS:
            if r[key]:
                rows.append([label, naira(r[key])])
        if r["annual_rent"]:
            rows.append(["Rent relief (20%% of %s rent, max %s)"
                         % (naira(r["annual_rent"]), naira(tax.RENT_RELIEF_CAP)),
                         naira(r["rent_relief"])])
        rows += [["Eligible deductions", naira(r["eligible_deductions"])],
                 ["Chargeable income", naira(r["chargeable_income"])]]
        rows += [["%s at %g%%" % (b["label"], b["rate"]), naira(b["tax"])]
                 for b in r["bands"] if b["applied"]]
        rows += [["Annual PAYE", naira(r["annual_tax"])],
                 ["Monthly PAYE", naira(r["monthly_tax"])]]
    elif tax_type == "cit":
        if r["is_small_company"]:
            reply = ("This is a small company (turnover %s or less and fixed assets %s or less), "
                     "so it pays no company income tax and no development levy."
                     % (naira(tax.SMALL_COMPANY_TURNOVER), naira(tax.SMALL_COMPANY_FIXED_ASSETS)))
        else:
            reply = ("Estimated company income tax: %s (%g%% of taxable profit), plus a "
                     "development levy of %s (%g%% of assessable profit) — %s in total."
                     % (naira(r["tax_payable"]), r["rate"], naira(r["development_levy"]),
                        r["development_levy_rate"], naira(r["total_payable"])))
            notes.append("This assumes a Nigerian-resident company; non-resident companies "
                         "don't pay the development levy.")
        if r["minimum_etr_may_apply"]:
            notes.append("With turnover of ₦50 billion or more, the 15% minimum effective tax "
                         "rate (s. 57) may also apply. It is worked out from audited accounts, "
                         "so it isn't included here.")
        rows = [["Annual turnover", naira(r["turnover"])]]
        if r["fixed_assets"] is not None:
            rows.append(["Total fixed assets", naira(r["fixed_assets"])])
        rows += [["Category", r["category"]],
                 ["Assessable profit", naira(r["assessable_profit"])],
                 ["Capital allowances", naira(r["capital_allowances"])],
                 ["Taxable profit", naira(r["taxable_profit"])],
                 ["CIT (%g%%)" % r["rate"], naira(r["tax_payable"])],
                 ["Development levy (%g%%)" % r["development_levy_rate"],
                  naira(r["development_levy"])],
                 ["Total payable", naira(r["total_payable"])]]
    elif tax_type == "vat":
        if r["mode"] == "inclusive":
            reply = ("%s already includes %s of VAT at %g%%; the amount before VAT is %s."
                     % (naira(r["total_amount"]), naira(r["vat_amount"]), r["rate"],
                        naira(r["net_amount"])))
        else:
            reply = ("VAT at %g%% on %s is %s, making %s in total."
                     % (r["rate"], naira(r["net_amount"]), naira(r["vat_amount"]),
                        naira(r["total_amount"])))
            notes.append("I treated the amount as excluding VAT. If it already includes "
                         "VAT, say \"inclusive\".")
        rows = [["Amount before VAT", naira(r["net_amount"])],
                ["VAT (%g%%)" % r["rate"], naira(r["vat_amount"])],
                ["Total including VAT", naira(r["total_amount"])]]
    else:
        reply = ("Withholding tax at %g%% on %s (%s): %s. The recipient receives %s."
                 % (r["rate"], naira(r["gross_amount"]), r["transaction_label"],
                    naira(r["wht_deducted"]), naira(r["net_amount"])))
        rows = [["Gross payment", naira(r["gross_amount"])],
                ["Payment type", r["transaction_label"]], ["Rate", "%g%%" % r["rate"]],
                ["WHT deducted", naira(r["wht_deducted"])],
                ["Net amount payable", naira(r["net_amount"])]]
    if notes:
        reply += "\n\n" + "\n".join(notes)
    return reply, rows


def naira(value: float) -> str:
    return "₦{:,.2f}".format(value)


# ── Routing ────────────────────────────────────────────────────────────────────

def classify(text: str) -> str:
    """'calculate', 'inform' or 'unclear' — decided by rules, in this order."""
    amounts, _ = find_amounts(text)
    if CALC_VERB.search(text):
        return "calculate"
    if QUESTION_START.match(text):
        return "inform"
    if HOW_MUCH.search(text) and (amounts or OWE.search(text)):
        return "calculate"
    trigger = TAX_ON.search(text)
    if trigger:
        follows_number = re.match(r"[₦\d]|n\d|ngn", text[trigger.end():], FLAGS)
        if amounts or not follows_number:
            return "calculate"
    if amounts:
        return "unclear"
    return "inform"


def _clarify(reply: str, pending: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "clarification", "reply": reply, "pending": pending}


def _resolve_type(types: set, generic: bool, text: str) -> Tuple[Optional[str], List[str]]:
    """Pick the tax to calculate, or the candidates to ask the user about."""
    if generic and not types & {"paye", "cit"}:
        paye_hint, cit_hint = bool(PAYE_HINT.search(text)), bool(CIT_HINT.search(text))
        if paye_hint != cit_hint:
            types = types | {"paye" if paye_hint else "cit"}
        else:
            return None, sorted(types | {"paye", "cit"})
    if len(types) == 1:
        return next(iter(types)), []
    if len(types) > 1:
        return None, sorted(types)
    paye_hint, cit_hint = bool(PAYE_HINT.search(text)), bool(CIT_HINT.search(text))
    if paye_hint != cit_hint:
        return ("paye" if paye_hint else "cit"), []
    return None, sorted(tax.CALCULATORS)


def _which_tax(candidates: List[str], message: str) -> Dict[str, Any]:
    names = ", ".join(TAX_NAMES[c] for c in candidates)
    return _clarify("Which tax should I calculate: %s?" % names,
                    {"choose_from": candidates, "message": message})


def _run(tax_type: str, text: str, known: Dict[str, Any]) -> Dict[str, Any]:
    """Collect slots for `tax_type` and calculate, or ask for what is missing."""
    _, _, masked = find_tax_types(text)
    amounts, saw_percent = find_amounts(text)
    found, problems = extract_slots(tax_type, text, masked, amounts, known)
    slots = dict(known, **found)
    pending = {"tax_type": tax_type, "slots": slots}
    if problems:
        return _clarify(" ".join(problems), pending)
    if _missing(tax_type, slots):
        return _clarify(_ask_for(tax_type, slots), pending)
    try:
        return calculate(tax_type, slots, saw_percent)
    except tax.TaxInputError as exc:
        return _clarify("I can't use that figure: %s Please give a corrected amount."
                        % _plain(str(exc)), pending)


def _plain(message: str) -> str:
    """tax.py speaks in field names; the chat speaks in words."""
    words = {"'gross_income'": "Gross income", "'amount'": "The amount",
             "'turnover'": "Turnover", "'assessable_profit'": "Assessable profit",
             "'capital_allowances'": "Capital allowances", "'pension'": "Pension",
             "'fixed_assets'": "Fixed assets", "'annual_rent'": "Rent", "'nhis'": "NHIS",
             "'nhf'": "NHF"}
    for field, word in words.items():
        message = message.replace(field, word)
    return message


EXAMPLES = {
    "paye": "Calculate PAYE on ₦500,000 monthly",
    "cit": "Calculate CIT: turnover ₦500m, profit ₦12m",
    "vat": "Calculate VAT on ₦100,000",
    "wht": "Calculate WHT on a ₦1m consultancy fee",
}


def _rate_summary(tax_type: str) -> str:
    """Describe the calculator's built-in rates, read from tax.py, never restated."""
    if tax_type == "vat":
        return "VAT at %g%%" % (tax.VAT_RATE * 100)
    if tax_type == "paye":
        rates = [rate * 100 for _, rate, _ in tax.PAYE_BANDS]
        return "PAYE on graduated bands from %g%% to %g%%" % (min(rates), max(rates))
    if tax_type == "cit":
        return ("company income tax at %g%% for small companies and %g%% otherwise, plus a "
                "%g%% development levy on companies that are not small"
                % (tax.CIT_SMALL_RATE * 100, tax.CIT_RATE * 100,
                   tax.DEVELOPMENT_LEVY_RATE * 100))
    rates = sorted({v["rate"] * 100 for v in tax.WHT_RATES.values()})
    return ("withholding tax at %s depending on the payment type"
            % " or ".join("%g%%" % r for r in rates))


def no_source_offer(types: set) -> str:
    """Reply for a question about a tax that the indexed passages don't cover.

    The corpus is the Act's definitions section, so "What is VAT?" finds nothing
    to cite. Answering from the model's memory would defeat the grounding rule,
    so instead say plainly what is missing and offer the calculator, describing
    its rates as built-in estimates rather than as statements of law.
    """
    ordered = [t for t in ("paye", "cit", "vat", "wht") if t in types]
    names = " or ".join(TAX_NAMES[t] for t in ordered)
    them = "it" if len(ordered) == 1 else "them"
    return (
        "The tax law passages I search don't explain %s, so I can't answer that with a "
        "source — and I won't answer from memory.\n\n"
        "I can calculate %s for you, though. The calculator uses built-in rates: %s. "
        "These are estimates and have not been independently verified against current law."
        "\n\nTry: \"%s\""
        % (names, them, "; ".join(_rate_summary(t) for t in ordered), EXAMPLES[ordered[0]])
    )


def _fresh(text: str) -> Dict[str, Any]:
    """Route a message that is not answering an earlier question."""
    intent = classify(text)
    types, generic, _ = find_tax_types(text)

    if intent == "inform":
        result = rag.answer_question(text)
        reply = result["answer"]
        named = types | ({"paye", "cit"} if generic else set())
        if not result["grounded"] and named:
            reply = no_source_offer(named)
        return {"type": "answer", "reply": reply, "sources": result["sources"],
                "grounded": result["grounded"], "pending": None}

    if intent == "unclear":
        amounts, _ = find_amounts(text)
        figure = naira(amounts[0]["value"])
        return _clarify(
            "Would you like a calculation using %s? If so, tell me which tax — for example "
            "\"Calculate VAT on %s\" or \"Calculate PAYE on %s monthly\". Or ask me a question "
            "about Nigerian tax." % (figure, figure, figure), None)

    unsupported = UNSUPPORTED.search(text)
    if unsupported and not types and not generic:
        return _clarify("I can't calculate %s. I can calculate PAYE, company income tax, "
                        "VAT and withholding tax — or answer a question about %s."
                        % (unsupported.group(0), unsupported.group(0)), None)

    tax_type, candidates = _resolve_type(types, generic, text)
    if tax_type is None:
        return _which_tax(candidates, text)
    return _run(tax_type, text, {})


def respond(message: str, pending: Any = None) -> Dict[str, Any]:
    """Reply to one chat message, continuing an unfinished calculation if any."""
    text = (message or "").strip()
    if not text:
        raise ValueError("Message must not be empty.")
    if len(text) > rag.MAX_QUESTION_CHARS:
        raise ValueError("Message is too long (limit %d characters)." % rag.MAX_QUESTION_CHARS)

    if CANCEL.match(text):
        return _clarify("OK, cancelled. Ask me a question or request a calculation.", None)

    state = validate_pending(pending)
    if state is None:
        return _fresh(text)

    if "choose_from" in state:
        types, generic, _ = find_tax_types(text)
        chosen = types & set(state["choose_from"])
        if not chosen and generic:
            chosen = {"paye", "cit"} & set(state["choose_from"])
        if not chosen:
            hint = {"paye"} if PAYE_HINT.search(text) else {"cit"} if CIT_HINT.search(text) else set()
            chosen = hint & set(state["choose_from"])
        if len(chosen) == 1:
            return _run(chosen.pop(), state["message"], {})
        return _fresh(text)

    # A new, explicit calculation request starts over rather than merging.
    tax_type = state["tax_type"]
    types, _, masked = find_tax_types(text)
    if classify(text) == "calculate" and types:
        return _fresh(text)
    if not _answers(tax_type, text, masked):
        return _fresh(text)
    return _run(tax_type, text, state["slots"])


def _answers(tax_type: str, text: str, masked: str) -> bool:
    """Does this message supply something the pending calculation needs?"""
    amounts, _ = find_amounts(text)
    if amounts:
        return True
    if tax_type == "paye":
        return bool(_period_mentions(masked))
    if tax_type == "vat":
        return bool(VAT_INCLUSIVE.search(masked) or VAT_EXCLUSIVE.search(masked))
    if tax_type == "wht":
        return any(p.search(masked) for p in WHT_TYPES.values())
    return False


def validate_pending(pending: Any) -> Optional[Dict[str, Any]]:
    """Accept a client-supplied pending object only if it is well formed.

    It came back from the browser, so it is untrusted: unknown keys are dropped
    and anything malformed discards the whole object (the message is then
    handled as a fresh request). Figures are re-checked by tax.py regardless.
    """
    if not isinstance(pending, dict):
        return None

    if "choose_from" in pending:
        choices, message = pending.get("choose_from"), pending.get("message")
        if (isinstance(choices, list) and choices and isinstance(message, str)
                and 0 < len(message) <= rag.MAX_QUESTION_CHARS
                and all(c in tax.CALCULATORS for c in choices)):
            return {"choose_from": list(dict.fromkeys(choices)), "message": message}
        return None

    tax_type, slots = pending.get("tax_type"), pending.get("slots")
    if tax_type not in tax.CALCULATORS or not isinstance(slots, dict):
        return None
    clean: Dict[str, Any] = {}
    for key, value in slots.items():
        if key in NUMERIC_SLOTS[tax_type]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                return None
            clean[key] = float(value)
        elif tax_type == "paye" and key in PERIOD_SLOTS:
            if value not in PERIODS:
                return None
            clean[key] = value
        elif tax_type == "vat" and key == "mode":
            if value not in ("exclusive", "inclusive"):
                return None
            clean[key] = value
        elif tax_type == "wht" and key == "transaction_type":
            if value not in tax.WHT_RATES:
                return None
            clean[key] = value
    return {"tax_type": tax_type, "slots": clean}
