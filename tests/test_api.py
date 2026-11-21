from fastapi.testclient import TestClient

from src import api


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda: None
    )

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda: None
    )

    expected_response = {
        "question": "How many days of annual leave?",
        "answer": "25 working days.",
        "references": [
            {
                "source": "hr_policy.txt",
                "section": "Annual Leave",
                "chunk_index": 1,
                "distance": 0.15
            }
        ]
    }

    monkeypatch.setattr(
        api,
        "ask_question",
        lambda question: {
            **expected_response,
            "question": question
        }
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "How many days of annual leave?"
            }
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "25 working days."
    assert response.json()["references"][0]["source"] == "hr_policy.txt"


def test_ask_rejects_short_questions(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda: None
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/ask",
            json={"question": "Hi"}
        )

    assert response.status_code == 422


def test_ask_rejects_missing_question(monkeypatch):
    monkeypatch.setattr(
        api,
        "ingest_documents",
        lambda: None
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/ask",
            json={}
        )

    assert response.status_code == 422
