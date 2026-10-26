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

"""MB Cloud's lightweight circuit intermediate representation.

The public shape is intentionally familiar to users of circuit SDKs, while the
implementation is focused on cloud transport and hardware compilation needs.
"""

from __future__ import annotations
import copy
import math
from numbers import Real
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from mb_client.common.parameters import Parameter, ParameterExpression, bind_value, parameters_in


@dataclass(frozen=True, order=True)
class Qubit:
    index: int


@dataclass(frozen=True, order=True)
class Clbit:
    index: int


@dataclass(frozen=True)
class CircuitInstruction:
    operation: "Instruction"
    qubits: tuple[Qubit, ...]
    clbits: tuple[Clbit, ...] = ()


class Instruction:
    """A circuit operation with fixed quantum/classical arity."""

    def __init__(self,
                 name: str,
                 num_qubits: int,
                 num_clbits: int = 0,
                 params: Sequence[object] = ()):
        if not name or num_qubits < 0 or num_clbits < 0:
            raise ValueError("Instruction name and bit counts must be valid")
        self.name = name
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.params = list(params)
        self._definition: QuantumCircuit | None = None

    @property
    def definition(self) -> "QuantumCircuit | None":
        return self._definition

    @definition.setter
    def definition(self, circuit: "QuantumCircuit | None") -> None:
        if circuit is not None and circuit.num_qubits != self.num_qubits:
            raise ValueError(
                "Instruction definition has the wrong number of qubits")
        self._definition = circuit

    def copy(self) -> "Instruction":
        return copy.deepcopy(self)


class Gate(Instruction):
    """An instruction without classical outputs."""

    def __init__(self,
                 name: str,
                 num_qubits: int,
                 params: Sequence[object] = ()):
        super().__init__(name, num_qubits, 0, params)


class QuantumCircuit:
    """Ordered logical operations over indexed quantum and classical bits."""

    def __init__(self,
                 num_qubits: int,
                 num_clbits: int = 0,
                 name: str | None = None,
                 *,
                 bias: Mapping[int, object] | None = None):
        if num_qubits < 0 or num_clbits < 0:
            raise ValueError("Circuit bit counts cannot be negative")
        self.name = name or "circuit"
        self.metadata: dict[str, object] = {}
        self.physical_qubits = tuple(range(num_qubits))
        self.qubits = tuple(Qubit(index) for index in range(num_qubits))
        self.clbits = tuple(Clbit(index) for index in range(num_clbits))
        self.data: list[CircuitInstruction] = []
        self._bias = self._validate_bias(bias or {})

    def _validate_bias(self, bias):
        result = {}
        for qubit, value in bias.items():
            if type(qubit) is not int or not 0 <= qubit < self.num_qubits:
                raise ValueError(f"invalid logical bias qubit {qubit!r}")
            if not isinstance(value, (Parameter, ParameterExpression)):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        "bias must be a finite real number or parameter")
                if not math.isfinite(value):
                    raise ValueError("bias must be a finite real number")
                value = float(value)
            result[qubit] = copy.deepcopy(value)
        return result

    @property
    def bias(self) -> dict[int, object]:
        """A copy of whole-execution bias settings keyed by logical qubit.

        Unspecified qubits use backend defaults. Bias has no timeline position.
        """
        return copy.deepcopy(self._bias)

    def with_bias(self, bias: Mapping[int, object]) -> "QuantumCircuit":
        """Return a copy with supplied settings replacing matching logical entries."""
        overrides = self._validate_bias(bias)
        target = copy.deepcopy(self)
        target._bias.update(overrides)
        return target

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def num_clbits(self) -> int:
        return len(self.clbits)

    @property
    def parameters(self):
        found = set(parameters_in(self._bias))
        for item in self.data:
            found.update(parameters_in(item.operation.params))
            found.update(
                parameters_in(getattr(item.operation, "calibration", {})))
            found.update(
                parameters_in(getattr(item.operation, "parameters", {})))
        return frozenset(found)

    def bind_parameters(self,
                        values,
                        *,
                        per_qubit_values=None,
                        inplace: bool = False) -> "QuantumCircuit":
        per_qubit_values = per_qubit_values or {}
        unknown = (set(values).union(per_qubit_values).difference(
            self.parameters).difference(parameter.name
                                        for parameter in self.parameters))
        if unknown:
            raise ValueError(
                f"Unknown circuit parameters: {sorted(map(str, unknown))}")
        target = self if inplace else copy.deepcopy(self)
        bound_bias = {}
        for qubit, value in target._bias.items():
            local_values = dict(values)
            for parameter in parameters_in(value):
                if parameter.name in values and parameter not in local_values:
                    if sum(p.name == parameter.name
                           for p in self.parameters) != 1:
                        raise ValueError(
                            f"Ambiguous parameter name {parameter.name!r}; use Parameter keys"
                        )
                    local_values[parameter] = values[parameter.name]
                qubit_values = per_qubit_values.get(
                    parameter, per_qubit_values.get(parameter.name))
                if qubit_values is not None:
                    physical = target.physical_qubits[qubit]
                    if physical not in qubit_values:
                        raise ValueError(
                            f"missing value for parameter {parameter.name!r} on physical qubit {physical}"
                        )
                    local_values[parameter] = qubit_values[physical]
            bound_bias[qubit] = bind_value(value, local_values)
        target._bias = target._validate_bias(bound_bias)
        for item in target.data:
            operation = item.operation
            item_parameters = set(parameters_in(operation.params))
            for attribute in ("calibration", "parameters"):
                item_parameters.update(
                    parameters_in(getattr(operation, attribute, {})))
            local_values = dict(values)
            # Circuit binding is explicit user input, unlike pulse operation binding.
            for parameter in item_parameters:
                if parameter.name in values and parameter not in local_values:
                    same_name = {
                        p
                        for p in self.parameters if p.name == parameter.name
                    }
                    if len(same_name) != 1:
                        raise ValueError(
                            f"Ambiguous parameter name {parameter.name!r}; use Parameter keys"
                        )
                    local_values[parameter] = values[parameter.name]
            for parameter, qubit_values in per_qubit_values.items():
                if parameter not in item_parameters:
                    continue
                if len(item.qubits) != 1:
                    raise ValueError(
                        f"per-qubit parameter {parameter.name!r} is used by "
                        f"multi-qubit instruction {operation.name!r}")
                physical = target.physical_qubits[item.qubits[0].index]
                if physical not in qubit_values:
                    raise ValueError(
                        f"missing value for parameter {parameter.name!r} on "
                        f"physical qubit {physical}")
                local_values[parameter] = qubit_values[physical]
            operation.params = bind_value(operation.params, local_values)
            for attribute in ("calibration", "parameters"):
                if hasattr(operation, attribute):
                    setattr(
                        operation, attribute,
                        bind_value(getattr(operation, attribute),
                                   local_values))
        return target

    def append(
        self,
        operation: Instruction,
        qubits: Sequence[int | Qubit],
        clbits: Sequence[int | Clbit] = ()
    ) -> CircuitInstruction:
        if not isinstance(operation, Instruction):
            raise TypeError("operation must be an MB Cloud Instruction")
        qargs = tuple(self._resolve_qubit(bit) for bit in qubits)
        cargs = tuple(self._resolve_clbit(bit) for bit in clbits)
        if len(qargs) != operation.num_qubits or len(
                cargs) != operation.num_clbits:
            raise ValueError(
                f"{operation.name} expects ({operation.num_qubits}, "
                f"{operation.num_clbits}) bits, got ({len(qargs)}, {len(cargs)})"
            )
        if len(set(qargs)) != len(qargs):
            raise ValueError("An instruction cannot use the same qubit twice")
        item = CircuitInstruction(operation.copy(), qargs, cargs)
        self.data.append(item)
        return item

    def compose(self,
                other: "QuantumCircuit",
                qubits: Sequence[int | Qubit] | None = None,
                inplace: bool = False) -> "QuantumCircuit":
        target = self if inplace else copy.deepcopy(self)
        mapped = tuple(target.qubits if qubits is None else (
            target._resolve_qubit(bit) for bit in qubits))
        if len(mapped) != other.num_qubits:
            raise ValueError("compose qubit mapping has the wrong size")
        if len(set(mapped)) != len(mapped):
            raise ValueError("compose qubit mapping contains duplicates")
        merged_bias = target.bias
        for qubit, value in other.bias.items():
            mapped_qubit = mapped[qubit].index
            if mapped_qubit in merged_bias and merged_bias[
                    mapped_qubit] != value:
                raise ValueError(
                    f"conflicting static biases for logical qubit {mapped_qubit}"
                )
            merged_bias[mapped_qubit] = value
        for item in other.data:
            target.append(item.operation,
                          [mapped[bit.index] for bit in item.qubits],
                          [bit.index for bit in item.clbits])
        target._bias = merged_bias
        return target

    def to_gate(self, name: str | None = None) -> Gate:
        if self._bias:
            raise ValueError("A circuit with static bias cannot become a Gate")
        if self.num_clbits:
            raise ValueError(
                "A circuit containing classical bits cannot become a Gate")
        gate = Gate(name or self.name, self.num_qubits)
        gate.definition = copy.deepcopy(self)
        return gate

    def barrier(self, *qubits: int | Qubit) -> CircuitInstruction:
        from mb_client.circuit.operations import Barrier
        qargs = qubits or self.qubits
        return self.append(Barrier(len(qargs)), qargs)

    def measure(self, qubit: int | Qubit | Sequence[int | Qubit],
                clbit: int | Clbit | Sequence[int | Clbit], **options):
        """Measure a scalar pair or equally sized lists of quantum/classical bits.

        List calls return a tuple and add no implicit alignment barrier.
        All operands are validated before any instructions are appended.
        """
        from mb_client.circuit.operations import Measure
        multiple = isinstance(qubit, (list, tuple))
        if multiple != isinstance(clbit, (list, tuple)):
            raise TypeError("measure requires two scalars or two lists/tuples")
        qubits = qubit if multiple else (qubit, )
        clbits = clbit if multiple else (clbit, )
        if len(qubits) != len(clbits):
            raise ValueError(
                "measurement qubits and clbits must have equal lengths")
        qargs = tuple(self._resolve_qubit(q) for q in qubits)
        cargs = tuple(self._resolve_clbit(c) for c in clbits)
        if len(set(qargs)) != len(qargs) or len(set(cargs)) != len(cargs):
            raise ValueError(
                "measurement operands must not contain duplicates")
        operation = Measure(**options)
        items = tuple(
            self.append(operation, [q], [c]) for q, c in zip(qargs, cargs))
        return items if multiple else items[0]

    def measure_all(self, **options) -> tuple[CircuitInstruction, ...]:
        """Measure logical qubit i into clbit i, adding missing classical bits.

        Existing bits and instructions are retained. No barrier is inserted.
        """
        if self.num_clbits < self.num_qubits:
            self.clbits += tuple(
                Clbit(i) for i in range(self.num_clbits, self.num_qubits))
        return self.measure(list(self.qubits),
                            list(self.clbits[:self.num_qubits]), **options)

    def delay(self,
              duration: float,
              qubit: int | Qubit,
              channel: str = "drive") -> CircuitInstruction:
        from mb_client.circuit.operations import Delay
        return self.append(Delay(duration, channel), [qubit])

    def pulse_block(self,
                    sequence,
                    qubits,
                    *,
                    name: str | None = None) -> CircuitInstruction:
        """Insert a copy of an explicitly engineered PulseSequence(template=True).

        Channel index i addresses qubits[i]. The compiler ordering policy controls
        synchronization with surrounding instructions. Delays and barriers inside
        the sequence control internal timing.
        """
        from mb_client.pulse.block import PulseBlock
        targets = (qubits, ) if isinstance(qubits,
                                           (int, Qubit)) else tuple(qubits)
        return self.append(PulseBlock(sequence, len(targets), name=name),
                           targets)

    def x(self, qubit, **calibration):
        from mb_client.circuit.gates import XGate
        return self.append(XGate(**calibration), [qubit])

    def y(self, qubit, **calibration):
        from mb_client.circuit.gates import YGate
        return self.append(YGate(**calibration), [qubit])

    def id(self, duration: float, qubit):
        from mb_client.circuit.gates import IGate
        return self.append(IGate(duration), [qubit])

    def sx(self, qubit, **calibration):
        from mb_client.circuit.gates import SXGate
        return self.append(SXGate(**calibration), [qubit])

    def sy(self, qubit, **calibration):
        from mb_client.circuit.gates import SYGate
        return self.append(SYGate(**calibration), [qubit])

    def unitary(self, matrix, qubits, *, name: str = "unitary"):
        from mb_client.circuit.gates import UnitaryGate
        if isinstance(qubits, (int, Qubit)):
            qubits = [qubits]
        return self.append(UnitaryGate(matrix, name=name), qubits)

    def rz(self, phi, qubit):
        from mb_client.circuit.gates import RZGate
        return self.append(RZGate(phi), [qubit])

    def rx(self, theta, qubit):
        from mb_client.circuit.gates import RXGate
        return self.append(RXGate(theta), [qubit])

    def ry(self, theta, qubit):
        from mb_client.circuit.gates import RYGate
        return self.append(RYGate(theta), [qubit])

    def rxy(self, theta, phi, qubit):
        from mb_client.circuit.gates import RXYGate
        return self.append(RXYGate(theta, phi), [qubit])

    def h(self, qubit):
        from mb_client.circuit.gates import HGate
        return self.append(HGate(), [qubit])

    def cx(self, control, target):
        from mb_client.circuit.gates import CXGate
        return self.append(CXGate(), [control, target])

    def cz(self, control, target):
        from mb_client.circuit.gates import CZGate
        return self.append(CZGate(), [control, target])

    def reset(self, *qubits, align: bool | None = None, **calibration):
        """Reset scalar, positional, or list operands; omit operands for all qubits.

        An aligned reset brackets the individual reset instructions with
        barriers.  The leading barrier makes every selected qubit start its
        reset at the same time; the trailing barrier prevents subsequent
        instructions from starting until every reset has finished.
        """
        from mb_client.circuit.operations import Reset
        if len(qubits) == 1 and isinstance(qubits[0], (list, tuple)):
            qubits = tuple(qubits[0])
        targets = tuple(
            self._resolve_qubit(q) for q in (qubits or tuple(self.qubits)))
        if len(set(targets)) != len(targets):
            raise ValueError("reset operands must not contain duplicates")
        if align is None:
            align = len(targets) > 1
        if align:
            self.barrier(*targets)
        resets = tuple(
            self.append(Reset(**calibration), [qubit]) for qubit in targets)
        if align:
            self.barrier(*targets)
        return resets[0] if len(resets) == 1 else resets

    def _resolve_qubit(self, bit: int | Qubit) -> Qubit:
        resolved = self.qubits[bit] if isinstance(bit, int) else bit
        if resolved not in self.qubits:
            raise ValueError(f"Qubit {bit!r} is not in this circuit")
        return resolved

    def _resolve_clbit(self, bit: int | Clbit) -> Clbit:
        resolved = self.clbits[bit] if isinstance(bit, int) else bit
        if resolved not in self.clbits:
            raise ValueError(f"Clbit {bit!r} is not in this circuit")
        return resolved

    def __iter__(self) -> Iterable[CircuitInstruction]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def draw(self, format: str = "text", style=None, *, show: bool = False):
        """Draw the circuit as text or a Matplotlib figure.

        Set ``show=True`` to emit the drawing immediately, which is useful for
        calls made inside a notebook loop.  The default keeps rendering free of
        display side effects, so assigned figures can be saved or composed.
        """
        from mb_client.visualization.circuit import draw_circuit
        drawing = draw_circuit(self, format=format, style=style)
        if show:
            self._display_drawing(drawing)
        return drawing

    @staticmethod
    def _display_drawing(drawing) -> None:
        """Display a drawing in IPython, with a terminal-safe fallback."""
        try:
            from IPython.display import display
        except ImportError:
            if isinstance(drawing, str):
                print(drawing)
            else:
                import matplotlib.pyplot as plt
                plt.show()
        else:
            display(drawing)

    def __str__(self) -> str:
        return self.draw(format="text")
