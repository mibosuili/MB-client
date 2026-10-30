"""Shared fixtures and test data for client tests."""

import os

import pytest

from mb_client import MBCloudClient


class FakeResponse:
    """Mock a requests response."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.content = b"json" if payload is not None else b""

    def json(self):
        return self.payload


class FakeSession:
    """In-memory Session that records requests and returns fixed results."""

    def __init__(self):
        self.headers = {}
        self.submissions = []

    def request(self, method, url, *, json, timeout):
        if url.endswith("/api/v1/system/health/"):
            return FakeResponse({"message": "pong"})
        if method == "POST" and url.endswith("/api/v1/jobs/"):
            self.submissions.append(json)
            return FakeResponse({"job_id": "job-1"})
        if url.endswith("/api/v1/jobs/job-1/"):
            return FakeResponse({
                "job_id": "job-1", "status": "completed", "error": None
            })
        if url.endswith("/api/v1/jobs/job-1/result/"):
            return FakeResponse({
                "job_id": "job-1",
                "experiment": "circuit",
                "device": "teaching-chip",
                "backend_name": "teaching-chip",
                "run_options": {"shots": 1000, "measure_level": "state"},
                "program_results": [{
                    "index": 0,
                    "input_index": 0,
                    "program_type": "circuit",
                    "name": "bell",
                    "physical_qubits": [1, 2],
                    "shots": 1000,
                    "measure_level": "state",
                    "data": {"00": 490, "11": 510},
                }],
                "metadata": {},
            })
        raise AssertionError(f"Unhandled test request: {method} {url}")


@pytest.fixture
def fake_session():
    """Create an isolated in-memory HTTP Session."""
    return FakeSession()


@pytest.fixture
def client(fake_session):
    """Create an SDK client that does not access the network."""
    return MBCloudClient(session=fake_session)


@pytest.fixture
def isolated_api_key_environment(monkeypatch, tmp_path):
    """Isolate client environment variables, .env, and the current directory."""
    monkeypatch.delenv("MB_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("MB_CLOUD_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    yield
    # The Dynaconf dotenv loader loads file values into the process environment.
    os.environ.pop("MB_CLOUD_API_KEY", None)
    os.environ.pop("MB_CLOUD_BASE_URL", None)
