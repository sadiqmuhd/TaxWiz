"""HTTP-level tests: routing, request validation and the JSON error contract."""

import pytest

import rag


# ── Health and pages ───────────────────────────────────────────────────────────

def test_health_reports_healthy(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "healthy"
    assert body["service"] == "TaxWiz"


def test_health_does_not_require_credentials(client, monkeypatch):
    """The probe must stay green even with no API keys, or Railway kills the app."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    body = client.get("/health").get_json()
    assert body["status"] == "healthy"
    assert body["openai_configured"] is False


def test_home_page_renders_the_ui(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"TaxWiz" in response.data
    # The frontend must call the endpoints this app actually exposes.
    assert b"/api/ask" in response.data
    assert b"/api/calculate/" in response.data


# ── Calculator endpoint ────────────────────────────────────────────────────────

def test_calculate_paye_endpoint(client):
    response = client.post("/api/calculate/paye", json={
        "gross_income": 1_200_000, "pension": 96_000, "nhf": 30_000,
    })

    assert response.status_code == 200
    body = response.get_json()
    assert body["annual_tax"] == 59_100
    assert body["monthly_tax"] == 4_925
    assert len(body["bands"]) == 3


@pytest.mark.parametrize("tax_type,payload", [
    ("cit", {"turnover": 50_000_000, "assessable_profit": 12_000_000}),
    ("vat", {"amount": 500_000, "mode": "exclusive"}),
    ("wht", {"amount": 1_000_000, "transaction_type": "contracts"}),
])
def test_other_calculators_respond(client, tax_type, payload):
    response = client.post("/api/calculate/%s" % tax_type, json=payload)

    assert response.status_code == 200
    assert response.get_json()["tax_type"] == tax_type


def test_calculate_rejects_bad_input_with_json_error(client):
    response = client.post("/api/calculate/paye", json={"gross_income": -100})

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Invalid input"
    assert "negative" in body["message"]


def test_calculate_rejects_missing_body(client):
    response = client.post("/api/calculate/vat")

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid request"


def test_calculate_rejects_unknown_tax_type(client):
    response = client.post("/api/calculate/inheritance", json={"amount": 1})

    assert response.status_code == 400
    assert "Unknown tax type" in response.get_json()["message"]


def test_calculate_rejects_wrong_method(client):
    response = client.get("/api/calculate/vat")

    assert response.status_code == 405
    assert response.get_json()["error"] == "Method not allowed"


def test_tax_types_endpoint_matches_the_engine(client):
    body = client.get("/api/tax-types").get_json()

    assert body["tax_types"] == ["cit", "paye", "vat", "wht"]
    assert body["vat_rate"] == 7.5
    assert body["wht_rates"]["consultancy"]["rate"] == 5.0


# ── Ask endpoint ───────────────────────────────────────────────────────────────

def test_ask_requires_a_question(client):
    response = client.post("/api/ask", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid request"


def test_ask_rejects_a_blank_question(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400


def test_ask_returns_the_answer_and_its_sources(client, monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda question: {
        "answer": "VAT is charged at 7.5%. [1]",
        "sources": [{"id": "chunk-1", "score": 0.71, "excerpt": "VAT rate..."}],
        "grounded": True,
    })

    body = client.post("/api/ask", json={"question": "What is VAT?"}).get_json()

    assert body["answer"].startswith("VAT is charged")
    assert body["sources"][0]["id"] == "chunk-1"


def test_ask_reports_missing_configuration_as_503(client, monkeypatch):
    def explode(question):
        raise rag.ConfigurationError("OPENAI_API_KEY is not configured.")

    monkeypatch.setattr(rag, "answer_question", explode)
    response = client.post("/api/ask", json={"question": "What is VAT?"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.get_json()["message"]


def test_ask_never_leaks_a_stack_trace(client, monkeypatch):
    def explode(question):
        raise RuntimeError("secret internal detail: token=abc123")

    monkeypatch.setattr(rag, "answer_question", explode)
    response = client.post("/api/ask", json={"question": "What is VAT?"})

    assert response.status_code == 500
    body = response.get_json()
    assert "abc123" not in str(body)
    assert "Traceback" not in str(body)


def test_unknown_api_route_returns_json_not_html(client):
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.get_json()["error"] == "Not found"
