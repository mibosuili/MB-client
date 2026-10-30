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

"""Mutable pulse implementations and calibrations with isolated snapshots."""
from __future__ import annotations

import copy
from threading import RLock
from typing import Any

import pandas as pd

from mb_client.circuit import Instruction
from mb_client.common.parameters import Parameter, ParameterExpression
from mb_client.pulse import PulseSequence
from mb_client.pulse.sequence import transform_sequence, sequence_parameters, _ensure_concrete
from .calibration import CalibrationError, CalibrationParams
from .schedule_map import InstSchedRegistry, _same


class Target:
    """Own pulse implementations and calibration values for physical qubits.

    Getters and constructor inputs are copied. Setters replace components;
    updates merge entries. Mutations are atomic and affect future compilations.
    """

    def __init__(self, *, inst_sched_reg: InstSchedRegistry,
                 calibration_params: CalibrationParams | None = None):
        self._lock = RLock()
        self.set_inst_sched_reg(inst_sched_reg)
        self.set_calibration_params(
            CalibrationParams() if calibration_params is None else calibration_params)

    @property
    def qubits(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._inst_sched_reg.qubits)

    def get_inst_sched_reg(self) -> InstSchedRegistry:
        with self._lock:
            return copy.deepcopy(self._inst_sched_reg)

    def get_calibration_params(self) -> CalibrationParams:
        with self._lock:
            return self._copy_params(self._calibration_params)

    def get_calibration_param(self, qubit: int | tuple[int, ...], parameter: str, *,
                              gate: str | None = None) -> Any:
        """Read a gate-specific value, falling back to the qubit-wide value.

        Template defaults are used only by resolve(), never by this getter.
        """
        qubits = (qubit,) if type(qubit) is int else qubit
        if (not isinstance(qubits, tuple) or not qubits
                or any(type(q) is not int or q < 0 for q in qubits)
                or len(set(qubits)) != len(qubits)):
            raise ValueError("qubit must be a non-negative integer or a tuple of unique non-negative integers")
        with self._lock:
            return self._calibration_params.get(qubits, parameter, gate=gate)

    @staticmethod
    def _copy_registry(registry):
        if not isinstance(registry, InstSchedRegistry):
            raise TypeError("registry must be InstSchedRegistry")
        # Recheck mutable public qubits and cross-entry parameter defaults.
        InstSchedRegistry(registry.qubits)
        candidate = copy.deepcopy(registry)
        for name, qubits in candidate._entries:
            candidate._key(name, qubits)
        candidate.default_calibration_params()
        return candidate

    @staticmethod
    def _copy_params(params):
        if not isinstance(params, CalibrationParams):
            raise TypeError("params must be CalibrationParams")
        return CalibrationParams(params.to_dataframe())

    def set_inst_sched_reg(self, registry: InstSchedRegistry) -> None:
        candidate = self._copy_registry(registry)
        with self._lock:
            self._inst_sched_reg = candidate

    def set_calibration_params(self, params: CalibrationParams) -> None:
        candidate = self._copy_params(params)
        with self._lock:
            self._calibration_params = candidate

    def update_inst_sched_reg(self, registry: InstSchedRegistry) -> None:
        patch = self._copy_registry(registry)
        with self._lock:
            candidate = copy.deepcopy(self._inst_sched_reg)
            candidate.qubits = list(dict.fromkeys([*candidate.qubits, *patch.qubits]))
            candidate._entries.update(patch._entries)
            candidate.default_calibration_params()
            self._inst_sched_reg = candidate

    def update_calibration_params(self, params: CalibrationParams) -> None:
        patch = self._copy_params(params)
        with self._lock:
            frames = [self._calibration_params.to_dataframe(), patch.to_dataframe()]
            nonempty = [frame for frame in frames if not frame.empty]
            candidate = (CalibrationParams(pd.concat(nonempty, ignore_index=True))
                         if nonempty else CalibrationParams())
            self._calibration_params = candidate

    def _snapshot(self) -> Target:
        """Capture both components under one lock for compilation/preparation."""
        with self._lock:
            return Target(inst_sched_reg=self._inst_sched_reg,
                          calibration_params=self._calibration_params)

    def resolve(self, instruction: Instruction, qubits: tuple[int, ...], *,
                cbits: tuple[int, ...] | None = None) -> PulseSequence:
        """Resolve pulse parameters and optionally bind classical destinations.

        Omit cbits to preserve template destinations. Currently only single-bit
        measurement instructions support classical operands.
        """
        if not isinstance(instruction, Instruction):
            raise TypeError("instruction must be Instruction")
        if cbits is not None:
            cbits = tuple(cbits)
            if len(cbits) != instruction.num_clbits:
                raise ValueError("classical operand count mismatch")
            if any(type(bit) is not int or bit < 0 for bit in cbits):
                raise ValueError("cbits must contain non-negative integers")
            if cbits and (instruction.name != 'measure' or len(cbits) != 1):
                raise NotImplementedError(
                    "classical operands are supported only for single-bit measurement")
        with self._lock:
            return self._resolve(instruction, qubits, cbits)

    def _resolve(self, gate, qubits, cbits=None):
        key = self._inst_sched_reg._key(gate, qubits)
        entry = self._inst_sched_reg._get_entry(gate, qubits)
        sequence = entry.sequence
        if len(gate.params) != len(entry.arguments):
            raise ValueError(f"argument count mismatch for {key}")
        bindings = {}
        for signature, actual in zip(entry.arguments, gate.params):
            if isinstance(signature, Parameter):
                bindings[signature.name] = actual
            elif not _same(signature, actual):
                raise ValueError(f"fixed gate argument mismatch for {key}")
        overrides = getattr(gate, "calibration", {})
        calibration_names = {
            p.name for p in sequence_parameters(sequence)
            if p.group == "calibration"
        }
        unknown = set(overrides) - calibration_names
        if unknown:
            raise ValueError(
                f"unknown calibration overrides for {key}: {unknown}; "
                f"valid calibration parameters: {sorted(calibration_names)}")

        def value(item):
            if isinstance(item, ParameterExpression):
                operands = tuple(value(x) for x in item.operands)
                if any(
                        isinstance(x, (Parameter, ParameterExpression))
                        for x in operands):
                    return ParameterExpression(item.function, operands,
                                               item.symbol)
                return item.function(*operands)
            if not isinstance(item, Parameter):
                return copy.deepcopy(item)
            if item.group == 'operation':
                return copy.deepcopy(bindings.get(item.name, item))
            if item.name in overrides and overrides[item.name] is not None:
                return copy.deepcopy(overrides[item.name])
            device, name, scope = self._inst_sched_reg._calibration_key(item, key)
            try:
                result = self._calibration_params.get(device, name, gate=scope)
            except CalibrationError:
                result = None
            if result is None:
                if item.has_default:
                    return copy.deepcopy(item.default)
                raise CalibrationError(
                    f'Missing calibration {name!r} for {key} on {device}')
            return result

        try:
            result = transform_sequence(
                sequence, value,
                clbit=cbits[0] if cbits else None)
            _ensure_concrete(result)
            return result
        except (ValueError, TypeError) as exc:
            raise type(exc)(f'{key}: {exc}') from exc
