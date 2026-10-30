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

"""Abstract job lifecycle shared by local and remote execution."""
from abc import ABC, abstractmethod
from enum import Enum
from .results import ExecutionResult


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobExecutionError(RuntimeError):
    """Execution failed; the original local exception is chained."""


class JobCancelledError(RuntimeError):
    """The job was cancelled before completion."""


class Job(ABC):
    @property
    @abstractmethod
    def job_id(self) -> str:
        ...

    @abstractmethod
    def status(self) -> JobStatus:
        ...

    @abstractmethod
    def cancel(self) -> bool:
        """Return whether cancellation was accepted."""

    @abstractmethod
    def result(self, timeout: float | None = None) -> ExecutionResult:
        """Wait for result. A timeout does not cancel execution."""

    @property
    def id(self) -> str:
        """Alias for job_id, following common SDK naming conventions."""
        return self.job_id


    wait = result
