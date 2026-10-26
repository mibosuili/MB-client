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

"""Unitary gates and their circuit decompositions."""

from __future__ import annotations
import numpy as np
from mb_client.circuit.circuit import Gate, QuantumCircuit


class IGate(Gate):

    def __init__(self, length: float | None = None, **calibration):
        self.calibration = dict(calibration)
        super().__init__("id", 1, [] if length is None else [length])


class XGate(Gate):

    def __init__(self, **calibration):
        self.calibration = dict(calibration)
        super().__init__("x", 1)


class YGate(Gate):

    def __init__(self, **calibration):
        self.calibration = dict(calibration)
        super().__init__("y", 1)


class SYGate(Gate):
    """Principal square root of Pauli Y, equivalent to ``RY(pi / 2)``."""

    def __init__(self, **calibration):
        self.calibration = dict(calibration)
        super().__init__("sy", 1)


class SXGate(Gate):

    def __init__(self, **calibration):
        self.calibration = dict(calibration)
        super().__init__("sx", 1)


class SXdgGate(Gate):

    def __init__(self, **calibration):
        self.calibration = dict(calibration)
        super().__init__("sxdg", 1)


class RXGate(Gate):

    def __init__(self, theta: float, **calibration):
        self.calibration = dict(calibration)
        super().__init__("rx", 1, [theta])


class RYGate(Gate):

    def __init__(self, theta: float, **calibration):
        self.calibration = dict(calibration)
        super().__init__("ry", 1, [theta])


class RXYGate(Gate):
    """Rotation by ``theta`` around an axis at ``phi`` in the XY plane."""

    def __init__(self, theta: float, phi: float, **calibration):
        self.calibration = dict(calibration)
        super().__init__("rxy", 1, [theta, phi])


class RZGate(Gate):

    def __init__(self, phi: float):
        super().__init__("rz", 1, [phi])


class HGate(Gate):

    def __init__(self):
        super().__init__("h", 1)
        definition = QuantumCircuit(1, name="h")
        definition.rz(np.pi / 2, 0)
        definition.sx(0)
        definition.rz(np.pi / 2, 0)
        self.definition = definition


class CXGate(Gate):

    def __init__(self):
        super().__init__("cx", 2)
        definition = QuantumCircuit(2, name="cx")
        definition.h(1)
        definition.cz(0, 1)
        definition.h(1)
        self.definition = definition


class CZGate(Gate):

    def __init__(self):
        super().__init__("cz", 2)


class UnitaryGate(Gate):
    """A numeric one- or two-qubit unitary operation."""

    def __init__(self, matrix, *, name: str = "unitary"):
        try:
            value = np.asarray(matrix, dtype=complex)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "unitary matrix must contain numeric values") from exc
        if value.shape not in ((2, 2), (4, 4)):
            raise ValueError("unitary matrix must have shape (2, 2) or (4, 4)")
        if not np.all(np.isfinite(value)):
            raise ValueError("unitary matrix must contain only finite values")
        identity = np.eye(value.shape[0], dtype=complex)
        if not np.allclose(
                value.conj().T @ value, identity, rtol=1e-9, atol=1e-10):
            raise ValueError("matrix is not unitary")
        self._matrix = value.copy()
        super().__init__(name, 1 if value.shape == (2, 2) else 2)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()


class VirtualZ(RZGate):
    """Compatibility name for a logical Z-frame rotation."""
