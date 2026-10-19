from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("IMAGE_SEARCH_API_KEY", "test-secret-key")
    monkeypatch.setenv("IMAGE_SEARCH_ALLOWED_ROOTS", str(Path.home()))
    return TestClient(main.app)


def test_api_requires_key(client):
    assert client.get("/status").status_code == 401


def test_api_rejects_wrong_key(client):
    assert client.get("/status", headers={"X-API-Key": "wrong"}).status_code == 401


def test_api_accepts_configured_key(client):
    response = client.get("/status", headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 200
    assert "image_count" in response.json()


def test_missing_server_key_fails_closed(client, monkeypatch):
    monkeypatch.delenv("IMAGE_SEARCH_API_KEY")
    response = client.get("/status", headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 503


def test_search_rejects_excessive_top_k(client):
    response = client.get(
        "/search?query=tree&top_k=1000",
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 422


def test_index_path_outside_allowed_roots_is_rejected(client, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    allowed = Path.home() / "image-search-test-allowed"
    monkeypatch.setenv("IMAGE_SEARCH_ALLOWED_ROOTS", str(allowed))
    response = client.post(
        "/index",
        headers={"X-API-Key": "test-secret-key"},
        json={"directory": str(outside)},
    )
    assert response.status_code == 403
