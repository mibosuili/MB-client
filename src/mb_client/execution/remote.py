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
from .backends import Backend, Programs, snapshot_programs
from .jobs import Job, JobStatus
from .options import RunOptions
from .results import ExecutionResult


class RemoteClient(ABC):
    """Adapter owns serialization, authentication, polling and error mapping.

    results raises TimeoutError, JobCancelledError or JobExecutionError.
    Services must preserve program order and circuit classical-bit mappings.
    """
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

    def run(self, programs: Programs, **run_options) -> RemoteJob:
        options = RunOptions().with_overrides(**run_options)
        inputs = snapshot_programs(programs)
        job_id = self._client.submit(inputs, backend_name=self._backend_name,
                                     run_options=options)
        return self.retrieve_job(job_id)

    def retrieve_job(self, job_id: str) -> RemoteJob:
        return RemoteJob(job_id, client=self._client)
