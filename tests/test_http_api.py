"""Tests for requests.Session calls and authentication headers."""

from unittest.mock import Mock

from mb_client import MBCloudClient


def test_http_api_ping():
    """Verify the ping route and API key request header."""
    response = Mock(ok=True, content=b"json", status_code=200)
    response.json.return_value = {"status": "ok", "message": "pong"}
    session = Mock()
    session.headers = {}
    session.request.return_value = response

    client = MBCloudClient(api_key="test-key", session=session)

    assert client.ping() == "pong"
    session.request.assert_called_once_with(
        "GET",
        "http://127.0.0.1:8000/api/v1/system/health/",
        json=None,
        timeout=30.0,
    )
    assert session.headers["X-API-Key"] == "test-key"
