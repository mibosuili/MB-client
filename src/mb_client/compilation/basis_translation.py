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

"""Circuit-to-circuit translation into a native unitary gate basis."""

from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares
from mb_client.circuit import Gate, QuantumCircuit
from mb_client.circuit.gates import CZGate, RZGate, SXGate, UnitaryGate, XGate

DEFAULT_BASIS_GATES = ("rz", "sx", "x", "cz")
_SUPPORTED_BASIS_GATES = frozenset(DEFAULT_BASIS_GATES)
_ANGLE_TOLERANCE = 1e-10
_SYNTHESIS_TOLERANCE = 2e-8


def translate_basis(
    circuit_or_gate: QuantumCircuit | Gate,
    *,
    basis_gates: Sequence[str] = DEFAULT_BASIS_GATES,
) -> QuantumCircuit:
    """Lower a circuit or gate without routing or circuit optimization.

    Non-gate instructions are copied unchanged. Numeric unitary equivalence is
    defined up to global phase.
    """
    basis = _validate_basis(basis_gates)
    if isinstance(circuit_or_gate, Gate):
        source = QuantumCircuit(circuit_or_gate.num_qubits,
                                name=circuit_or_gate.name)
        source.append(circuit_or_gate, range(circuit_or_gate.num_qubits))
    elif isinstance(circuit_or_gate, QuantumCircuit):
        source = circuit_or_gate
    else:
        raise TypeError("translate_basis expects a QuantumCircuit or Gate")

    target = QuantumCircuit(source.num_qubits, source.num_clbits, source.name)
    target = target.with_bias(source.bias)
    target.metadata = dict(source.metadata)
    target.physical_qubits = tuple(source.physical_qubits)
    for item in source.data:
        qargs = tuple(qubit.index for qubit in item.qubits)
        cargs = tuple(clbit.index for clbit in item.clbits)
        if not isinstance(item.operation, Gate):
            target.append(item.operation, qargs, cargs)
            continue
        _lower_gate(target, item.operation, qargs, basis, ())
    return target


def _validate_basis(basis_gates: Sequence[str]) -> frozenset[str]:
    names = tuple(basis_gates)
    if len(names) != len(set(names)):
        raise ValueError("basis_gates must not contain duplicate names")
    unknown = set(names).difference(_SUPPORTED_BASIS_GATES)
    if unknown:
        raise ValueError(f"unsupported basis gates: {sorted(unknown)}")
    missing = {"rz", "sx", "cz"}.difference(names)
    if missing:
        raise ValueError(
            f"basis_gates is missing required gates: {sorted(missing)}")
    return frozenset(names)


def _lower_gate(target: QuantumCircuit, gate: Gate, qargs: tuple[int, ...],
                basis: frozenset[str], stack: tuple[str, ...]) -> None:
    if gate.name in basis:
        target.append(gate, qargs)
        return

    if gate.name in stack:
        raise ValueError(f"Cyclic gate decomposition involving {gate.name!r}")

    if isinstance(gate, UnitaryGate):
        synthesized = (_synthesize_one_qubit(gate.matrix, basis)
                       if gate.num_qubits == 1 else _synthesize_two_qubit(
                           gate.matrix, basis))
        _append_mapped(target, synthesized, qargs, basis, (*stack, gate.name))
        return

    if gate.definition is not None:
        _append_mapped(target, gate.definition, qargs, basis,
                       (*stack, gate.name))
        return

    # Named rotations have no definitions because their parameters are the
    # definition. Lower those explicitly before reporting an opaque gate.
    if gate.name == "y":
        _append_y(target, qargs[0], basis, half=False)
    elif gate.name == "sy":
        _append_y(target, qargs[0], basis, half=True)
    elif gate.name == "sxdg":
        _append_rx(target, -math.pi / 2, qargs[0], basis)
    elif gate.name == "rx":
        _append_rx(target, _numeric_angle(gate), qargs[0], basis)
    elif gate.name == "ry":
        _append_ry(target, _numeric_angle(gate), qargs[0], basis)
    elif gate.name == "rxy":
        theta, phi = map(float, gate.params)
        _append_rz(target, -phi, qargs[0])
        _append_rx(target, theta, qargs[0], basis)
        _append_rz(target, phi, qargs[0])
    elif gate.name == "x":
        _append_x(target, qargs[0], basis)
    elif gate.name == "id":
        # A timed identity is a scheduling instruction, not a removable unitary.
        # Keep it so lowering preserves its all-channel synchronization semantics.
        if gate.params:
            target.append(gate, qargs)
        return
    else:
        raise ValueError(
            f"Cannot translate_basis opaque non-basis gate {gate.name!r}: "
            "provide a definition or UnitaryGate")


def _append_mapped(target: QuantumCircuit, source: QuantumCircuit,
                   parent_qargs: tuple[int, ...], basis: frozenset[str],
                   stack: tuple[str, ...]) -> None:
    for item in source.data:
        qargs = tuple(parent_qargs[qubit.index] for qubit in item.qubits)
        if isinstance(item.operation, Gate):
            _lower_gate(target, item.operation, qargs, basis, stack)
        else:
            cargs = tuple(clbit.index for clbit in item.clbits)
            target.append(item.operation, qargs, cargs)


def _numeric_angle(gate: Gate) -> float:
    try:
        return float(gate.params[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gate {gate.name!r} has an unbound angle") from exc


def _append_rz(circuit: QuantumCircuit, angle: float, qubit: int) -> None:
    angle = _canonical_angle(angle)
    if not math.isclose(angle, 0.0, abs_tol=_ANGLE_TOLERANCE):
        circuit.append(RZGate(angle), [qubit])


def _append_x(circuit: QuantumCircuit, qubit: int,
              basis: frozenset[str]) -> None:
    if "x" in basis:
        circuit.append(XGate(), [qubit])
    else:
        circuit.append(SXGate(), [qubit])
        circuit.append(SXGate(), [qubit])


def _append_y(circuit: QuantumCircuit,
              qubit: int,
              basis: frozenset[str],
              *,
              half: bool,
              inverse: bool = False) -> None:
    sign = -1.0 if inverse else 1.0
    _append_rz(circuit, -sign * math.pi / 2, qubit)
    if half:
        if inverse:
            _append_rx(circuit, -math.pi / 2, qubit, basis)
        else:
            circuit.append(SXGate(), [qubit])
    else:
        _append_x(circuit, qubit, basis)
    _append_rz(circuit, sign * math.pi / 2, qubit)


def _append_ry(circuit: QuantumCircuit, angle: float, qubit: int,
               basis: frozenset[str]) -> None:
    angle = _canonical_angle(angle)
    if math.isclose(angle, math.pi, abs_tol=_ANGLE_TOLERANCE):
        _append_y(circuit, qubit, basis, half=False)
    elif math.isclose(angle, math.pi / 2, abs_tol=_ANGLE_TOLERANCE):
        _append_y(circuit, qubit, basis, half=True)
    elif math.isclose(angle, -math.pi / 2, abs_tol=_ANGLE_TOLERANCE):
        _append_y(circuit, qubit, basis, half=True, inverse=True)
    elif not math.isclose(angle, 0.0, abs_tol=_ANGLE_TOLERANCE):
        _append_rz(circuit, -math.pi / 2, qubit)
        _append_rx(circuit, angle, qubit, basis)
        _append_rz(circuit, math.pi / 2, qubit)


def _append_rx(circuit: QuantumCircuit, angle: float, qubit: int,
               basis: frozenset[str]) -> None:
    angle = _canonical_angle(angle)
    if math.isclose(abs(angle), math.pi, abs_tol=_ANGLE_TOLERANCE):
        _append_x(circuit, qubit, basis)
        if angle < 0:
            # X and -X differ only by global phase.
            return
    elif math.isclose(angle, math.pi / 2, abs_tol=_ANGLE_TOLERANCE):
        circuit.append(SXGate(), [qubit])
    elif not math.isclose(angle, 0.0, abs_tol=_ANGLE_TOLERANCE):
        _append_rz(circuit, math.pi / 2, qubit)
        circuit.append(SXGate(), [qubit])
        _append_rz(circuit, math.pi + angle, qubit)
        circuit.append(SXGate(), [qubit])
        _append_rz(circuit, math.pi / 2, qubit)


def _synthesize_one_qubit(matrix: np.ndarray,
                          basis: frozenset[str]) -> QuantumCircuit:
    circuit = QuantumCircuit(1)
    # Prefer exact named rotations, both for readability and stable output.
    if _equivalent(matrix, _X):
        _append_x(circuit, 0, basis)
        return circuit
    if _equivalent(matrix, _Y):
        _append_y(circuit, 0, basis, half=False)
        return circuit
    if _equivalent(matrix, _SX):
        circuit.append(SXGate(), [0])
        return circuit
    if _equivalent(matrix, _SY):
        _append_y(circuit, 0, basis, half=True)
        return circuit

    theta = 2 * math.atan2(abs(matrix[1, 0]), abs(matrix[0, 0]))
    if math.isclose(math.sin(theta / 2), 0.0, abs_tol=_ANGLE_TOLERANCE):
        phi = 0.0
        lam = np.angle(matrix[1, 1]) - np.angle(matrix[0, 0])
    else:
        phi = np.angle(matrix[1, 0]) - np.angle(matrix[0, 0])
        lam = np.angle(-matrix[0, 1]) - np.angle(matrix[0, 0])
    # Circuit order is the reverse of matrix multiplication order.
    _append_rz(circuit, lam, 0)
    _append_ry(circuit, theta, 0, basis)
    _append_rz(circuit, phi, 0)
    return circuit


def _synthesize_two_qubit(matrix: np.ndarray,
                          basis: frozenset[str]) -> QuantumCircuit:
    """Fit the universal four-local-layer, three-CZ Cartan template."""
    starts = [np.zeros(24)]
    generator = np.random.default_rng(8675309)
    starts.extend(generator.uniform(-math.pi, math.pi, 24) for _ in range(4))
    best = None
    for start in starts:
        result = least_squares(_two_qubit_residual,
                               start,
                               args=(matrix, ),
                               max_nfev=4000,
                               ftol=1e-12,
                               xtol=1e-12,
                               gtol=1e-12)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(
                best.fun):
            best = result
        if np.linalg.norm(result.fun) < _SYNTHESIS_TOLERANCE:
            break
    if best is None or np.linalg.norm(best.fun) >= _SYNTHESIS_TOLERANCE:
        raise ValueError("two-qubit unitary synthesis did not converge")

    circuit = QuantumCircuit(2)
    layers = best.x.reshape(4, 2, 3)
    for layer_index, layer in enumerate(layers):
        for qubit, (phi, theta, lam) in enumerate(layer):
            _append_rz(circuit, lam, qubit)
            _append_ry(circuit, theta, qubit, basis)
            _append_rz(circuit, phi, qubit)
        if layer_index != 3:
            circuit.append(CZGate(), [0, 1])
    return circuit


def _two_qubit_residual(parameters: np.ndarray,
                        target: np.ndarray) -> np.ndarray:
    candidate = _two_qubit_template(parameters)
    overlap = np.vdot(target.ravel(), candidate.ravel())
    phase = overlap / abs(overlap) if abs(overlap) > 1e-15 else 1.0
    difference = candidate - phase * target
    return np.concatenate((difference.real.ravel(), difference.imag.ravel()))


def _two_qubit_template(parameters: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=complex)
    layers = parameters.reshape(4, 2, 3)
    for index, layer in enumerate(layers):
        local = np.kron(_u3(*layer[0]), _u3(*layer[1]))
        result = local @ result
        if index != 3:
            result = _CZ @ result
    return result


def _u3(phi: float, theta: float, lam: float) -> np.ndarray:
    return _rz(phi) @ _ry(theta) @ _rz(lam)


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def _ry(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def _canonical_angle(angle: float) -> float:
    value = (float(angle) + math.pi) % (2 * math.pi) - math.pi
    return math.pi if math.isclose(value, -math.pi,
                                   abs_tol=_ANGLE_TOLERANCE) else value


def _equivalent(first: np.ndarray, second: np.ndarray) -> bool:
    overlap = np.vdot(second.ravel(), first.ravel())
    if abs(overlap) <= 1e-15:
        return False
    phase = overlap / abs(overlap)
    return np.allclose(first, phase * second, rtol=1e-9, atol=1e-10)


_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
_SY = _ry(math.pi / 2)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
