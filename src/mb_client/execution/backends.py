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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BackendInfo:
    """Immutable native capabilities, independent of pulse calibrations.

    Qubits are physical IDs. Coupling edges are directed and indicate support
    for at least one native two-qubit gate, not support for every basis gate.
    None means all-to-all connectivity; an empty tuple means no connections.
    Simulator status is independent of connectivity: simulators may be restricted.
    Measurement, reset and barriers are not basis gates.
    """

    basis_gates: tuple[str, ...]
    qubits: tuple[int, ...]
    coupling_map: tuple[tuple[int, int], ...] | None
    is_simulator: bool

    def __post_init__(self):
        if type(self.is_simulator) is not bool:
            raise ValueError("is_simulator must be a boolean")
        for name in ('basis_gates', 'qubits', 'coupling_map'):
            value = getattr(self, name)
            if name == "coupling_map" and value is None:
                continue
            if not isinstance(value, (list, tuple)):
                raise ValueError(f'{name} must be a list or tuple')
            object.__setattr__(self, name, tuple(value))
        if any(not isinstance(g, str) or not g.strip() for g in self.basis_gates):
            raise ValueError('basis_gates must contain nonempty gate names')
        if len(set(self.basis_gates)) != len(self.basis_gates):
            raise ValueError('basis_gates must be unique')
        if any(type(q) is not int or q < 0 for q in self.qubits):
            raise ValueError('qubits must contain nonnegative integer IDs')
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError('qubits must be unique')
        if self.coupling_map is None:
            return
        edges = []
        for edge in self.coupling_map:
            if (not isinstance(edge, (list, tuple)) or len(edge) != 2
                    or any(type(q) is not int or q not in self.qubits for q in edge)
                    or edge[0] == edge[1]):
                raise ValueError('coupling_map edges must join distinct IDs in qubits')
            edges.append(tuple(edge))
        object.__setattr__(self, 'coupling_map', tuple(edges))


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
    def info(self) -> BackendInfo:
        """Native capabilities; unsupported backends raise NotImplementedError."""
        raise NotImplementedError('Backend does not expose capability metadata')

    @property
    def basis_gates(self) -> tuple[str, ...]:
        """Native gate names, excluding measurement, reset and barriers."""
        return self.info.basis_gates

    @property
    def qubits(self) -> tuple[int, ...]:
        """Physical qubit IDs, which need not be consecutive."""
        return self.info.qubits

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def coupling_map(self) -> tuple[tuple[int, int], ...] | None:
        """Directed edges, None for all-to-all, or an empty tuple for none."""
        return self.info.coupling_map

    @property
    def is_simulator(self) -> bool:
        """Whether this backend executes on a simulator rather than hardware."""
        return self.info.is_simulator

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