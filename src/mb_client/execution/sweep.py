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

"""One- and two-dimensional parameter sweep definitions."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterable
from mb_client.common.parameters import Parameter
import copy
from mb_client.common import parameters_in
from mb_client.circuit import QuantumCircuit
from mb_client.pulse.schedule import PulseSchedule


class _SweepValues(dict):
    """Read-only record of the values used to bind a program."""

    def _reject_update(self, *args, **kwargs):
        raise TypeError("sweep metadata values cannot be updated; expand the template again")

    __setitem__ = __delitem__ = update = clear = pop = popitem = setdefault = __ior__ = _reject_update

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(dict(self), memo))


class SweepValueMode(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    DELTA = "delta"


@dataclass(frozen=True)
class SweepAxis:
    parameter: Parameter
    values: tuple[object, ...] | dict[int, tuple[object, ...]]
    value_mode: SweepValueMode

    def __init__(self,
                 parameter: Parameter,
                 values: Iterable[object] | Mapping[int, Iterable[object]],
                 *,
                 value_mode: SweepValueMode | str = SweepValueMode.ABSOLUTE):
        if not isinstance(parameter, Parameter):
            raise TypeError("sweep axis parameter must be a Parameter")
        try:
            value_mode = SweepValueMode(value_mode)
        except ValueError as exc:
            raise ValueError(
                f"unknown sweep value mode {value_mode!r}") from exc
        if (value_mode is not SweepValueMode.ABSOLUTE
                and parameter.group != "calibration"):
            raise ValueError(
                "relative and delta sweeps require a calibration parameter")
        if isinstance(values, Mapping):
            normalized = {}
            for qubit, qubit_values in values.items():
                if (isinstance(qubit, bool) or not isinstance(qubit, int)
                        or qubit < 0):
                    raise ValueError(
                        "per-qubit sweep keys must be non-negative integers")
                row = tuple(qubit_values)
                if not row:
                    raise ValueError(
                        f"sweep values for qubit {qubit} cannot be empty")
                normalized[qubit] = row
            if not normalized:
                raise ValueError("per-qubit sweep values cannot be empty")
            lengths = {len(row) for row in normalized.values()}
            if len(lengths) != 1:
                raise ValueError(
                    "per-qubit sweep values must have equal lengths")
            values = normalized
        else:
            values = tuple(values)
            if not values:
                raise ValueError("sweep axis values cannot be empty")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "value_mode", value_mode)

    @property
    def is_per_qubit(self) -> bool:
        return isinstance(self.values, Mapping)

    def __len__(self) -> int:
        if self.is_per_qubit:
            return len(next(iter(self.values.values())))
        return len(self.values)

    def value_at(self, index: int):
        if self.is_per_qubit:
            return {
                qubit: values[index]
                for qubit, values in self.values.items()
            }
        return self.values[index]


class SweepMode(Enum):
    GRID = "grid"
    ZIP = "zip"


@dataclass(frozen=True)
class Sweep:
    mode: SweepMode
    axes: tuple[SweepAxis, ...]

    def __post_init__(self):
        if not 1 <= len(self.axes) <= 2:
            raise ValueError("a sweep must contain one or two axes")
        parameters = [axis.parameter for axis in self.axes]
        if len(set(parameters)) != len(parameters):
            raise ValueError("sweep axes must use different parameters")
        if self.mode is SweepMode.ZIP:
            lengths = {len(axis) for axis in self.axes}
            if len(lengths) != 1:
                raise ValueError("zip sweep axes must have equal lengths")

    @classmethod
    def grid(cls, *axes: SweepAxis) -> "Sweep":
        return cls(SweepMode.GRID, tuple(axes))

    @classmethod
    def zip(cls, *axes: SweepAxis) -> "Sweep":
        return cls(SweepMode.ZIP, tuple(axes))

    @property
    def shape(self) -> tuple[int, ...]:
        if self.mode is SweepMode.GRID:
            return tuple(len(axis) for axis in self.axes)
        return (len(self.axes[0]), )

    def expand(self, circuits, *, target=None):
        """Return a list of bound programs.

        Sweep values are stored in metadata["sweep"], alongside copied template metadata.

        Accept one circuit, one PulseSchedule, or an iterable of either.
        Schedule start times can be symbolic and are bound explicitly; duration
        changes do not automatically move later entries.
        Relative/delta schedule sweeps use the parameter name and gate/qubits
        selectors to retrieve baselines; gate provenance is not inferred. The target is snapshotted once for
        calibration baselines; no compilation or backend submission is done.
        Execution pipelines should retain BatchPreparation so compilation can
        reuse its same target snapshot.
        """
        if isinstance(circuits, (QuantumCircuit, PulseSchedule)):
            circuits = [circuits]
        return BatchPreparation(target).expand(circuits, self)

    def points(self):
        axis_values = ([axis.value_at(index) for index in range(len(axis))]
                       for axis in self.axes)
        value_rows = (product(*axis_values)
                      if self.mode is SweepMode.GRID else zip(*axis_values))
        for flat_index, row in enumerate(value_rows):
            if self.mode is SweepMode.GRID:
                indices = _unravel(flat_index, self.shape)
            else:
                indices = (flat_index, ) * len(self.axes)
            yield ({
                axis.parameter: value
                for axis, value in zip(self.axes, row)
            }, indices)


def _unravel(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    indices = []
    for size in reversed(shape):
        indices.append(index % size)
        index //= size
    return tuple(reversed(indices))


class BatchPreparation:

    def __init__(self, target=None):
        from mb_client.compilation import Target
        if target is not None and not isinstance(target, Target):
            raise TypeError("target must be Target")
        self.target = target._snapshot() if target is not None else None

    def expand(self, inputs, sweep):
        """Prepare a batch using this object's fixed target snapshot."""
        inputs = list(inputs)
        if sweep is None:
            return copy.deepcopy(inputs)
        if not isinstance(sweep, Sweep):
            raise TypeError("sweep must be Sweep or None")

        baselines = []
        for batch_index, circuit in enumerate(inputs):
            self._validate_circuit(circuit, sweep, batch_index)
            baselines.append(self._prepare_schedule_sweep(circuit, sweep)
                             if isinstance(circuit, PulseSchedule)
                             else self._prepare_sweep(circuit, sweep))

        expanded = []
        # Generate in result order directly: sweep point first, then batch.
        for bindings, indices in sweep.points():
            for batch_index, circuit in enumerate(inputs):
                bound, row = self._bind_point(
                    circuit, sweep, bindings, indices,
                    baselines[batch_index], batch_index)
                expanded.append(bound)
                previous = bound.metadata.get("sweep", {})
                if not isinstance(previous, Mapping):
                    raise ValueError('template metadata["sweep"] must be a mapping')
                conflicts = set(previous).intersection(row)
                if conflicts:
                    raise ValueError(f"cannot overwrite template sweep values: {sorted(conflicts)}")
                bound.metadata["sweep"] = _SweepValues(copy.deepcopy({**previous, **row}))
        return expanded

    @staticmethod
    def _validate_circuit(circuit, sweep, batch_index):
        if not isinstance(circuit, (QuantumCircuit, PulseSchedule)):
            raise TypeError(f"sweep input {batch_index} must be QuantumCircuit or PulseSchedule")
        missing = {axis.parameter for axis in sweep.axes} - circuit.parameters
        if missing:
            names = sorted(parameter.name for parameter in missing)
            raise ValueError(
                f"Sweep parameters are not in circuit {batch_index}: {names}")

    @staticmethod
    def _bind_point(circuit, sweep, bindings, indices, baselines, batch_index):
        """Resolve one point's values and bind an independent circuit copy."""
        shared = {}
        per_qubit = {}
        row = {}
        for axis in sweep.axes:
            supplied = bindings[axis.parameter]
            if axis.is_per_qubit:
                supplied_by_qubit = supplied
            elif axis.value_mode is SweepValueMode.ABSOLUTE:
                shared[axis.parameter] = supplied
                row[axis.parameter.name] = supplied
                continue
            else:
                supplied_by_qubit = {
                    qubit: supplied
                    for qubit in circuit.physical_qubits
                }
                row[f"{axis.parameter.name}_input"] = supplied

            resolved = {}
            for qubit, input_value in supplied_by_qubit.items():
                if axis.value_mode is SweepValueMode.ABSOLUTE:
                    value = input_value
                elif axis.value_mode is SweepValueMode.RELATIVE:
                    value = (baselines[axis.parameter][qubit] *
                             input_value)
                else:
                    value = (baselines[axis.parameter][qubit] +
                             input_value)
                resolved[qubit] = value
                if axis.is_per_qubit:
                    row[f"{axis.parameter.name}_input_q{qubit}"] = input_value
                row[f"{axis.parameter.name}_q{qubit}"] = value
            per_qubit[axis.parameter] = resolved
        bound = circuit.bind_parameters(shared,
                                        per_qubit_values=per_qubit)
        if bound.parameters:
            names = sorted(parameter.name
                           for parameter in bound.parameters)
            raise ValueError(
                f"Unbound circuit parameters: {names}")
        return bound, row

    def _prepare_schedule_sweep(self, program, sweep):
        """Schedules have no gate provenance; relative baselines use selectors."""
        baselines = {}
        for axis in sweep.axes:
            if axis.is_per_qubit and set(axis.values) != set(program.physical_qubits):
                raise ValueError("per-qubit sweep mapping does not match schedule physical qubits")
            if axis.value_mode is SweepValueMode.ABSOLUTE:
                continue
            if self.target is None:
                raise ValueError("relative and delta sweeps require a target")
            baselines[axis.parameter] = {
                qubit: self.target.get_calibration_param(
                    axis.parameter.qubits or (qubit,), axis.parameter.name,
                    gate=axis.parameter.gate)
                for qubit in program.physical_qubits
            }
        return baselines

    def _prepare_sweep(self, circuit: QuantumCircuit, sweep: Sweep):
        """Validate qubit mappings and capture immutable calibration baselines."""
        expected = set(circuit.physical_qubits)
        baselines = {}
        for axis in sweep.axes:
            if axis.is_per_qubit:
                actual = set(axis.values)
                if actual != expected:
                    missing = sorted(expected - actual)
                    unexpected = sorted(actual - expected)
                    raise ValueError(
                        "per-qubit sweep mapping does not match circuit physical "
                        f"qubits; missing={missing}, unexpected={unexpected}")
            if axis.value_mode is SweepValueMode.ABSOLUTE:
                continue
            if self.target is None:
                raise ValueError("relative and delta sweeps require a target")
            parameter_baselines = {}
            for logical_index, physical_index in enumerate(
                    circuit.physical_qubits):
                calibration_keys = []
                if axis.parameter in parameters_in(
                        circuit.bias.get(logical_index)):
                    # Static settings resolve against qubit-level calibration.
                    calibration_keys.append((None, axis.parameter.name))
                for item in circuit.data:
                    operation = item.operation
                    occurrence_parameters = set(parameters_in(
                        operation.params))
                    for attribute in ("calibration", "parameters"):
                        occurrence_parameters.update(
                            parameters_in(getattr(operation, attribute, {})))
                    if axis.parameter not in occurrence_parameters:
                        continue
                    if len(item.qubits) != 1:
                        raise ValueError(
                            f"per-qubit parameter {axis.parameter.name!r} is used "
                            f"by multi-qubit instruction {operation.name!r}")
                    if item.qubits[0].index == logical_index:
                        fields = [
                            name
                            for attribute in ("calibration", "parameters")
                            for name, value in getattr(operation, attribute, {}).items()
                            if axis.parameter in parameters_in(value)
                        ]
                        # The destination field identifies the calibration;
                        # the symbolic parameter may have an independent name.
                        calibration_keys.extend(
                            (operation.name, name)
                            for name in (fields or [axis.parameter.name])
                        )
                if not calibration_keys:
                    raise ValueError(
                        f"parameter {axis.parameter.name!r} is not used on physical "
                        f"qubit {physical_index}")
                values = [
                    self.target.get_calibration_param(axis.parameter.qubits or physical_index,
                                               name,
                                               gate=gate)
                    for gate, name in calibration_keys
                ]
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(
                        f"parameter {axis.parameter.name!r} resolves to conflicting "
                        f"calibrations on physical qubit {physical_index}; use "
                        "separate Parameter objects")
                parameter_baselines[physical_index] = values[0]
            baselines[axis.parameter] = parameter_baselines
        return baselines
