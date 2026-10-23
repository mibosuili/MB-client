# Copyright 2026 mibosuili
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified entry point for the MB Cloud SDK."""

from collections.abc import Sequence
import time
from typing import Any

import requests

from .conf.settings import settings
from .exceptions import ServerConnectionError
from .execution import (ExecutionResult, JobCancelledError, JobExecutionError,
                        JobStatus, MeasurementLevel, Programs, RemoteBackend,
                        RemoteClient, RemoteJob, RunOptions)
from .circuit import QuantumCircuit
from .protocol import decode_result, encode_circuit


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


ROUTES = {
    "system.ping": ("GET", "/api/v1/system/health/"),
    "resource.list": ("GET", "/api/v1/resources/"),
    "user.me": ("GET", "/api/v1/user/me/"),
    "experiment.submit": ("POST", "/api/v1/jobs/"),
    "job.status": ("GET", "/api/v1/jobs/{job_id}/"),
    "job.result": ("GET", "/api/v1/jobs/{job_id}/result/"),
    "job.cancel": ("POST", "/api/v1/jobs/{job_id}/cancel/"),
}


class MBCloudClient(RemoteClient):
    """Client for submitting remote quantum experiments."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ):
        self.base_url = (
            base_url or settings.get("base_url", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.api_key = api_key or settings.get("api_key")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key

    def _request(self, operation: str, payload: dict[str, Any]) -> Any:
        """Call a server endpoint permitted by the SDK allowlist."""
        try:
            method, path_template = ROUTES[operation]
        except KeyError as exc:
            raise ServerConnectionError(
                f"Unsupported operation: {operation}"
            ) from exc
        path = path_template.format(**payload)
        body = None
        if method != "GET":
            body = {key: value for key, value in payload.items()
                    if key != "job_id"}
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                json=body,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ServerConnectionError(
                f"Cannot communicate with {self.base_url}: {exc}"
            ) from exc
        if not response.ok:
            try:
                error = response.json()
                detail = error.get("error", error)
                message = detail.get("message") or detail.get("detail")
            except (ValueError, AttributeError):
                message = None
            raise ServerConnectionError(
                message or f"HTTP {response.status_code}"
            )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise ServerConnectionError("Server returned invalid JSON") from exc

    def ping(self) -> str:
        return self._request("system.ping", {})["message"]

    def resources(self) -> list[dict]:
        """List available chips and their qubits."""
        return self._request("resource.list", {})

    def profile(self) -> dict:
        """Return the authenticated user's available machine time."""
        return self._request("user.me", {})

    def backend(self, device: str):
        """Return a device-bound circuit backend."""
        if not device:
            raise ValueError("device must not be empty")
        return RemoteBackend(self, backend_name=device)

    def run(self,
            programs: QuantumCircuit | Sequence[QuantumCircuit],
            *,
            device: str,
            shots: int = 1024,
            measure_level: MeasurementLevel | str = MeasurementLevel.STATE,
            metadata: dict | None = None) -> RemoteJob:
        """Submit one circuit or a homogeneous circuit batch to MB Cloud."""
        if isinstance(programs, QuantumCircuit):
            batch = (programs,)
        elif isinstance(programs, Sequence) and not isinstance(programs,
                                                                (str, bytes)):
            batch = tuple(programs)
        else:
            raise TypeError("programs must be a QuantumCircuit or sequence")
        if not batch or any(not isinstance(item, QuantumCircuit) for item in batch):
            raise ValueError("program batch must contain QuantumCircuit objects")
        backend = self.backend(device)
        job_id = self.submit(
            batch,
            backend_name=device,
            run_options=RunOptions(shots=shots, measure_level=measure_level),
            metadata=metadata,
        )
        return backend.retrieve_job(job_id)

    def submit(self, programs: Programs, *, backend_name: str,
               run_options: RunOptions,
               metadata: dict | None = None) -> str:
        """Submit a generic remote Backend request to the MB Cloud API."""
        if run_options.iq_centers is not None:
            raise ValueError("IQ centers for remote execution are provided by the server-side chip configuration")
        if any(not isinstance(program, QuantumCircuit) for program in programs):
            raise TypeError("MB Cloud currently supports only QuantumCircuit for remote execution")
        response = self._request("experiment.submit", {
            "experiment": "circuit",
            "device": backend_name,
            "programs": [encode_circuit(item) for item in programs],
            "run_options": {
                "shots": run_options.shots,
                "measure_level": run_options.measure_level.value,
            },
            "metadata": metadata or {},
        })
        return str(response["job_id"])

    def status(self, job_id: str) -> JobStatus:
        """Query the remote job status and map it to the unified status enum."""
        value = self.details(job_id)["status"]
        if value == "submitting":
            value = JobStatus.QUEUED.value
        return JobStatus(value)

    def details(self, job_id: str) -> dict:
        """Return the complete status information for a remote job."""
        return self._request("job.status", {"job_id": job_id})

    def cancel(self, job_id: str) -> bool:
        """Cancel a remote job that is still queued."""
        response = self._request("job.cancel", {"job_id": job_id})
        return bool(response["cancelled"])

    def results(self, job_id: str, *, timeout: float | None = 300.0,
                polling_interval: float = 0.25) -> ExecutionResult:
        """Poll a remote job and return a unified execution result."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        if polling_interval <= 0:
            raise ValueError("polling_interval must be positive")
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            details = self.details(job_id)
            status = details["status"]
            if status == JobStatus.COMPLETED.value:
                payload = self._request("job.result", {"job_id": job_id})
                return decode_result(payload)
            if status == JobStatus.FAILED.value:
                raise JobExecutionError(
                    details.get("error") or f"Job {job_id} failed"
                )
            if status == JobStatus.CANCELLED.value:
                raise JobCancelledError(job_id)
            time.sleep(polling_interval)
        raise TimeoutError(f"Job {job_id} is still pending")
