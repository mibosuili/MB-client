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

"""Backend interface and homogeneous program-batch validation."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
import copy
from typing import Any, Mapping, TYPE_CHECKING
from mb_client.common import Parameter
from mb_client.circuit import QuantumCircuit
from mb_client.pulse import PulseSchedule
from .jobs import Job

if TYPE_CHECKING:
    from mb_client.compilation import Target

Program = QuantumCircuit | PulseSchedule
Programs = Sequence[QuantumCircuit] | Sequence[PulseSchedule]
ParameterBinds = Mapping[Parameter | str, Any]


def snapshot_programs(programs: Programs) -> tuple[Program, ...]:
    if isinstance(programs, (QuantumCircuit, PulseSchedule, str, bytes)) or not isinstance(programs, Sequence):
        raise TypeError("backend.run requires a sequence of programs")
    inputs = tuple(programs)
    if not inputs:
        raise ValueError("program batch cannot be empty")
    if not (all(isinstance(p, QuantumCircuit) for p in inputs)
            or all(isinstance(p, PulseSchedule) for p in inputs)):
        raise TypeError("programs must be a homogeneous batch of QuantumCircuit or PulseSchedule")
    return copy.deepcopy(inputs)


class Backend(ABC):

    @property
    def target(self) -> "Target | None":
        """Compilation target, or None when the backend does not expose one."""
        return getattr(self, "_target", None)

    @target.setter
    def target(self, value: "Target | None") -> None:
        self._target = value

    @abstractmethod
    def run(self, programs: Programs, **run_options) -> Job:
        """Submit circuits or schedules; mixing the two is not supported."""