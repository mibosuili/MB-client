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

"""Circuit instruction containing an operand-relative pulse sequence."""

from __future__ import annotations
import copy
from dataclasses import replace
from mb_client.circuit import Instruction
from mb_client.common.parameters import bind_value, parameters_in


class PulseBlock(Instruction):
    """An owned PulseSequence with local channels numbered by circuit operand.

    Parameters are explicit circuit inputs, including calibration parameters:
    bind them before compilation. No schedule-map registration is consulted.
    Acquisitions retain physical readout-channel identity, not classical bits.
    """

    def __init__(self, sequence, num_qubits: int, *, name: str | None = None):
        from mb_client.pulse import PulseSequence, Barrier
        from mb_client.pulse.sequence import sequence_parameters

        if type(num_qubits) is not int or num_qubits < 1:
            raise ValueError("PulseBlock requires at least one qubit")
        if not isinstance(sequence, PulseSequence):
            raise TypeError("sequence must be PulseSequence")
        if not sequence.template:
            raise ValueError(
                "PulseBlock requires PulseSequence(template=True)")
        if sequence.bias_overrides:
            raise ValueError(
                "static bias belongs to the circuit, not a PulseBlock")
        for operation in sequence:
            channels = operation.channels if isinstance(
                operation, Barrier) else (operation.channel, )
            if any(channel.qubit_index >= num_qubits for channel in channels):
                raise ValueError(
                    "pulse channel operand is outside the block qubits")
        self._sequence = copy.deepcopy(sequence)
        self._arguments = tuple(dict.fromkeys(sequence_parameters(sequence)))
        super().__init__(name or "pulse_block",
                         num_qubits,
                         params=self._arguments)

    @property
    def sequence(self):
        """Return an independent operand-relative sequence with current bindings."""
        from mb_client.pulse import PulseSequence
        from mb_client.pulse.sequence import transform_sequence

        if len(self.params) != len(self._arguments):
            raise ValueError("PulseBlock parameter arity mismatch")
        bindings = dict(zip(self._arguments, self.params))
        resolved = transform_sequence(
            self._sequence,
            transform_value=lambda value: bind_value(value, bindings))
        return PulseSequence(resolved, template=True, metadata=resolved.metadata)

    def lower(self, qubits):
        """Copy and map channels; the compiler policy controls block boundaries."""
        from .sequence import (
            PulseSequence,
            Barrier,
            DriveChannel,
            FluxChannel,
            ReadoutChannel,
        )
        from mb_client.pulse.sequence import _ensure_concrete
        from mb_client.pulse.sequence import transform_sequence

        if len(qubits) != self.num_qubits:
            raise ValueError("PulseBlock qubit arity mismatch")
        if parameters_in(self.params):
            raise ValueError("unbound parameters in PulseBlock")
        boundary = tuple(
            channel(q) for q in qubits
            for channel in (DriveChannel, FluxChannel, ReadoutChannel))
        sequence = transform_sequence(self.sequence,
                                      transform_channel=lambda ch: type(ch)
                                      (qubits[ch.qubit_index]))
        result = PulseSequence(metadata=sequence.metadata)
        for operation in sequence:
            # An operand-free barrier belongs to this block, not other qubits.
            if isinstance(operation, Barrier) and not operation.channels:
                operation = replace(operation, channels=boundary)
            result.append(operation)
        _ensure_concrete(result)
        return result
