import pytest
from main_pipeline import quantify_opinion, ingest_article

def test_quantify_opinion():
    sample_text = "Remote work offers flexibility, but long hours may lead to burnout."
    result = quantify_opinion(sample_text)
    for key in ["work_flexibility", "burnout_risk", "remote_work_appeal", "productivity_impact", "overall_sentiment"]:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], int), f"Value for {key} is not an integer."
        assert 1 <= result[key] <= 5, f"Value for {key} out of range: {result[key]}"

def test_ingest_article(monkeypatch):
    # Simulate a dummy HTML response for article ingestion
    class DummyResponse:
        text = "<html><body><p>This is a test article.</p></body></html>"
    def dummy_get(url):
        return DummyResponse()
    monkeypatch.setattr("requests.get", dummy_get)
    article = ingest_article("http://dummy.url")
    assert "This is a test article." in article
