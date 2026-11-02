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

"""Cloud execution through an injected service client"""
from abc import ABC, abstractmethod
from threading import Lock
from .backends import Backend, BackendInfo, Programs, snapshot_programs
from .jobs import Job, JobStatus
from .options import RunOptions
from .results import ExecutionResult


class RemoteClient(ABC):
    """Adapter owns serialization, authentication, polling and error mapping.

    results raises TimeoutError, JobCancelledError or JobExecutionError.
    Services must preserve program order and circuit classical-bit mappings.
    """
    def backend_info(self, *, backend_name: str) -> BackendInfo:
        """Fetch native capabilities, when supported by the service adapter."""
        raise NotImplementedError('Remote client does not expose capability metadata')

    @abstractmethod
    def submit(self, programs: Programs, *, backend_name: str,
               run_options: RunOptions) -> str:
        ...

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        ...

    @abstractmethod
    def details(self, job_id: str) -> dict:
        ...

    @abstractmethod
    def results(self, job_id: str, *, timeout: float | None = 300.0,
                polling_interval: float = 0.25) -> ExecutionResult:
        ...


class RemoteJob(Job):
    def __init__(self, job_id: str, *, client: RemoteClient):
        self._job_id = job_id
        self._client = client
        self._result = None
        self._lock = Lock()

    @property
    def job_id(self) -> str:
        return self._job_id

    def status(self) -> JobStatus:
        with self._lock:
            if self._result is not None:
                return JobStatus.COMPLETED
        return JobStatus(self._client.status(self.job_id))

    def cancel(self) -> bool:
        with self._lock:
            if self._result is not None:
                return False
        return self._client.cancel(self.job_id)

    def details(self) -> dict:
        """Return cloud metadata such as queue position, priority, and execution time."""
        return self._client.details(self.job_id)

    def results(self, timeout: float | None = 300.0) -> ExecutionResult:
        with self._lock:
            if self._result is not None:
                return self._result
        result = self._client.results(self.job_id, timeout=timeout)
        if not isinstance(result, ExecutionResult) or result.job_id != self.job_id:
            raise ValueError("Cloud result has an unexpected type or job_id")
        with self._lock:
            if self._result is None:
                self._result = result
            return self._result

    def wait(self, timeout: float | None = 300.0,
             polling_interval: float = 0.25) -> ExecutionResult:
        with self._lock:
            if self._result is not None:
                return self._result
        result = self._client.results(
            self.job_id,
            timeout=timeout,
            polling_interval=polling_interval,
        )
        with self._lock:
            if self._result is None:
                self._result = result
            return self._result

    result = wait


class RemoteBackend(Backend):
    def __init__(self, client: RemoteClient, *, backend_name: str):
        self._client = client
        self._backend_name = backend_name
        self._info: BackendInfo | None = None
        self._info_lock = Lock()

    def _fetch_info(self) -> BackendInfo:
        info = self._client.backend_info(backend_name=self._backend_name)
        if not isinstance(info, BackendInfo):
            raise TypeError('backend_info must return BackendInfo')
        return info

    @property
    def info(self) -> BackendInfo:
        """Lazily fetch and cache a complete immutable capability snapshot."""
        with self._info_lock:
            if self._info is None:
                self._info = self._fetch_info()
            return self._info

    def refresh_info(self) -> BackendInfo:
        """Reload capabilities; a failed fetch preserves the previous snapshot."""
        with self._info_lock:
            self._info = self._fetch_info()
            return self._info

    def run(self, programs: Programs, **run_options) -> RemoteJob:
        options = RunOptions().with_overrides(**run_options)
        inputs = snapshot_programs(programs)
        job_id = self._client.submit(inputs, backend_name=self._backend_name,
                                     run_options=options)
        return self.retrieve_job(job_id)

    def retrieve_job(self, job_id: str) -> RemoteJob:
        return RemoteJob(job_id, client=self._client)
