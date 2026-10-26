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

"""Per-program and batch execution results."""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Literal, Mapping
import pandas as pd
from .options import MeasurementLevel, RunOptions


@dataclass(frozen=True)
class ProgramResult:
    index: int
    input_index: int
    program_type: Literal["circuit", "pulse_schedule"]
    name: str | None
    physical_qubits: tuple[int, ...]
    shots: int
    measure_level: MeasurementLevel
    data: Any
    metadata: Mapping[str, Any]
    runtime_info: Mapping[str, Any]

    def __post_init__(self):
        for field in ("metadata", "runtime_info"):
            object.__setattr__(self, field,
                               copy.deepcopy(dict(getattr(self, field))))

    @property
    def coordinates(self) -> Mapping[str, Any]:
        """Return the sweep values retained in program metadata."""
        return copy.deepcopy(dict(self.metadata.get("sweep", {})))

    @property
    def counts(self) -> dict[str, int] | None:
        """Return the counts dictionary for state measurements."""
        return self.data if self.measure_level == MeasurementLevel.STATE else None


@dataclass(frozen=True)
class ExecutionResult:
    job_id: str
    backend_name: str
    experiment_name: str | None
    metadata: Mapping[str, Any]
    run_options: RunOptions
    program_results: tuple[ProgramResult, ...]
    runtime_info: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "program_results",
                           tuple(self.program_results))
        for field in ("metadata", "runtime_info"):
            object.__setattr__(self, field,
                               copy.deepcopy(dict(getattr(self, field))))

    @property
    def coordinates(self) -> pd.DataFrame:
        return pd.DataFrame(
            [copy.deepcopy(dict(p.coordinates)) for p in self.program_results])
