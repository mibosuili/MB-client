"""Tests for client environment configuration."""

import pytest

from mb_client import MBCloudClient
from mb_client.conf.settings import settings


pytestmark = pytest.mark.usefixtures("isolated_api_key_environment")


def test_environment_is_loaded_from_dotenv(tmp_path):
    """Verify that the SDK reads the API key and service URL from .env."""
    (tmp_path / ".env").write_text(
        "MB_CLOUD_API_KEY=file-key\n"
        "MB_CLOUD_BASE_URL=https://cloud.example.com\n"
    )
    settings.reload()

    client = MBCloudClient()

    assert client.api_key == "file-key"
    assert client.base_url == "https://cloud.example.com"


def test_explicit_values_override_environment(monkeypatch):
    """Verify that constructor arguments take precedence over environment variables."""
    monkeypatch.setenv("MB_CLOUD_API_KEY", "environment-key")
    monkeypatch.setenv("MB_CLOUD_BASE_URL", "https://env.example.com")
    settings.reload()

    client = MBCloudClient(
        base_url="https://cloud.mbslabs.tech",
        api_key="explicit-key",
    )

    assert client.api_key == "explicit-key"
    assert client.base_url == "https://cloud.mbslabs.tech"


def test_default_connection_values():
    """Verify that the local service URL is used when environment variables are not set."""
    settings.reload()

    client = MBCloudClient(timeout=10)

    assert client.base_url == "http://127.0.0.1:8000"
    assert client.timeout == 10
