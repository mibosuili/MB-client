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

"""compilation: schedule map APIs."""

from __future__ import annotations
import copy
from dataclasses import dataclass
import numbers
import numpy as np
import pandas as pd
from mb_client.compilation.calibration import CalibrationError, CalibrationParams
from mb_client.circuit import Instruction, XGate, SXGate, RZGate, Measure, Reset
from mb_client.common.parameters import Parameter
from mb_client.pulse import PulseSequence, Pulse, Drag, NetZero, DriveChannel, ReadoutChannel, FluxChannel, ErrorFunction, Idle
from mb_client.pulse.sequence import transform_sequence, sequence_parameters


def default_x_sx_pulse_template():
    duration = Parameter("duration", group="calibration")
    detuning = Parameter("detuning", group="calibration")
    beta = Parameter("beta", group="calibration")
    amplitude = Parameter("amplitude", group="calibration")
    frequency = Parameter("qubit_frequency", group="calibration")
    x_sx_template = PulseSequence(template=True)
    x_sx_template.play(
        DriveChannel(0),
        Pulse(Drag(duration, amplitude, beta, detuning), frequency=frequency))
    return x_sx_template


def default_rz_pulse_template(phi):
    rz_template = PulseSequence(template=True)
    rz_template.shift_phase(DriveChannel(0), phi)
    return rz_template


def default_fast_reset_pulse_template():
    duration = Parameter("duration", group="calibration")
    amplitude = Parameter("amplitude", group="calibration")
    sigma = Parameter("sigma", default=10e-9, group="calibration")
    post_pad = Parameter("post_pad", group="calibration")
    fast_reset_template = PulseSequence(template=True)
    fast_reset_template.play(FluxChannel(0),
                             Pulse(NetZero(duration, amplitude, sigma)))
    fast_reset_template.barrier(FluxChannel(0), DriveChannel(0))
    fast_reset_template.play(DriveChannel(0), Pulse(Idle(post_pad)))
    fast_reset_template.barrier(FluxChannel(0), DriveChannel(0))
    return fast_reset_template


def default_measure_pulse_template():
    delay = Parameter("delay", default=104e-9, group="calibration")
    duration = Parameter("duration", group="calibration")
    amplitude = Parameter("amplitude", group="calibration")
    sigma = Parameter("sigma", default=1e-9, group="calibration")
    frequency = Parameter("cavity_frequency", group="calibration")
    digitizer_delay = Parameter("digitizer_delay",
                                default=0,
                                group="calibration")
    measure_template = PulseSequence(template=True)
    measure_template.delay(ReadoutChannel(0), delay)
    measure_template.play(
        ReadoutChannel(0),
        Pulse(ErrorFunction(duration, amplitude, sigma), frequency=frequency))
    measure_template.acquire(ReadoutChannel(0),
                             duration,
                             demod_freq=frequency,
                             digitizer_delay=digitizer_delay)
    return measure_template


def default_inst_sched_registry(qubits):
    inst_map = InstSchedRegistry(qubits=qubits)
    reset = default_fast_reset_pulse_template()
    x_sx = default_x_sx_pulse_template()
    measure = default_measure_pulse_template()
    phi = Parameter("phi", group="operation")
    rz = default_rz_pulse_template(phi)

    for q in qubits:
        inst_map.add(Reset(), (q, ), reset)
        inst_map.add(XGate(), (q, ), x_sx)
        inst_map.add(SXGate(), (q, ), x_sx)
        inst_map.add(RZGate(phi), (q,), rz)
        inst_map.add(Measure(), (q, ), measure)
    return inst_map


def _same(a, b):
    return bool(np.array_equal(a, b))


def _validate_defaults(parameters):
    seen = {}
    for p in parameters:
        if p in seen:
            prior = seen[p]
            if prior.has_default != p.has_default or (
                    p.has_default and not _same(prior.default, p.default)):
                raise ValueError(f'Conflicting defaults for {p}')
        seen[p] = p


@dataclass(frozen=True)
class _ScheduleEntry:
    sequence: PulseSequence
    arguments: tuple[object, ...]


class InstSchedRegistry:
    """Register symbolic instruction sequences independently of calibrations.

    ``add(RZGate(phi), (1,), sequence)`` infers the operation signature.
    Template sequences use zero-based operands; stored sequences use physical IDs.
    """

    def __init__(self, qubits: list[int]):
        if not isinstance(qubits, list) or any(
                type(q) is not int or q < 0 for q in qubits):
            raise ValueError('qubits must be a list of non-negative integers')
        if len(set(qubits)) != len(qubits):
            raise ValueError('qubits must be distinct')
        self.qubits = qubits.copy()
        self._entries = {}

    def _key(self, gate, qubits):
        if not isinstance(gate, (str, Instruction)):
            raise TypeError('gate must be a name or Instruction')
        name = gate if isinstance(gate, str) else gate.name
        if not name or not name.strip():
            raise ValueError('gate name must be non-empty')
        if (not isinstance(qubits, tuple) or not qubits or any(
                type(q) is not int or q not in self.qubits for q in qubits)
                or len(set(qubits)) != len(qubits)):
            raise ValueError(f'invalid physical-qubit tuple {qubits!r}')
        if isinstance(gate, Instruction) and gate.num_qubits != len(qubits):
            raise ValueError(f'gate {name} arity does not match {qubits}')
        return name, qubits

    def add(self, gate, qubits, sequence):
        key = self._key(gate, qubits)
        if key in self._entries:
            raise ValueError(f'{key} already registered; use set() to replace')
        self.set(gate, qubits, sequence)

    def set(self, gate, qubits, sequence):
        key = self._key(gate, qubits)
        if not isinstance(sequence, PulseSequence):
            raise TypeError('sequence must be PulseSequence')
        arguments = tuple(gate.params) if isinstance(gate, Instruction) else ()
        declared = []
        for arg in arguments:
            if isinstance(arg, Parameter) and arg.group == 'operation':
                declared.append(arg)
            elif not isinstance(arg, (str, numbers.Real)) or (isinstance(
                    arg, numbers.Real) and not np.isfinite(arg)):
                raise ValueError(
                    'gate arguments must be bare operation parameters or finite numbers'
                )
        if len({p.name for p in declared}) != len(declared):
            raise ValueError('operation parameter names must be unique')
        parameters = sequence_parameters(sequence)
        _validate_defaults(parameters)
        if any(p.group == 'operation' and p not in declared
               for p in parameters):
            raise ValueError(
                'sequence contains undeclared operation parameters')
        for p in parameters:
            if p.group == 'calibration':
                self._calibration_key(p, key)

        def transform_channel(ch):
            index = ch.qubit_index
            if sequence.template:
                if index >= len(qubits):
                    raise ValueError(f'operand {index} out of range for {key}')
                return type(ch)(qubits[index])
            if index not in qubits:
                raise ValueError(f'channel {ch} does not belong to {key}')
            return ch

        stored = transform_sequence(copy.deepcopy(sequence),
                                    transform_channel=transform_channel)
        entry = _ScheduleEntry(stored, copy.deepcopy(arguments))
        # Validate cross-entry defaults before committing the replacement.
        entries = {**self._entries, key: entry}
        self._default_table(entries)
        self._entries[key] = entry

    def has(self, gate, qubits):
        """Whether an exact gate/qubit registration exists (without binding it)."""
        return self._key(gate, qubits) in self._entries

    def remove(self, gate, qubits):
        del self._entries[self._key(gate, qubits)]

    def get(self, gate, qubits) -> PulseSequence:
        """Return an independent symbolic sequence without binding arguments.

        An Instruction selects its registration by name and qubits, just like a
        string. Use Target.resolve() to bind its arguments and calibrations.
        """
        return copy.deepcopy(self._get_entry(gate, qubits).sequence)

    def _get_entry(self, gate, qubits) -> _ScheduleEntry:
        key = self._key(gate, qubits)
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(f'No PulseSequence for {key}') from None

    @staticmethod
    def _calibration_key(p, key):
        name, qubits = key
        selected = p.qubits if p.qubits is not None else qubits
        return selected, p.name, p.gate or name

    def _default_table(self, entries):
        rows = {}
        defaults = {}
        for key, entry in sorted(entries.items(),
                                 key=lambda x: (x[0][1], x[0][0])):
            for p in sequence_parameters(entry.sequence):
                if p.group != 'calibration':
                    continue
                target = self._calibration_key(p, key)
                if target in defaults:
                    old = defaults[target]
                    if old.has_default != p.has_default or (
                            p.has_default
                            and not _same(old.default, p.default)):
                        raise ValueError(f'Conflicting defaults for {target}')
                defaults[target] = p
                device, name, gate = target
                rows[target] = dict(
                    qubit=device,
                    parameter=name,
                    gate=gate,
                    value=copy.deepcopy(p.default) if p.has_default else None,
                    metadata=dict(needs_calibration=True,
                                  has_default=p.has_default))
        frame = pd.DataFrame(list(rows.values()),
                             columns=CalibrationParams.COLUMNS,
                             dtype=object)
        return CalibrationParams(frame)

    def default_calibration_params(self):
        """Generate the current parameter inventory with declared defaults."""
        return self._default_table(self._entries)

    def __len__(self):
        return len(self._entries)

    def __str__(self):
        lines = [
            f'InstSchedRegistry: qubits={self.qubits}, {len(self)} registrations'
        ]
        for (name, qubits), entry in sorted(self._entries.items(),
                                            key=lambda x: (x[0][1], x[0][0])):
            args = '(' + ', '.join(map(
                str, entry.arguments)) + ')' if entry.arguments else ''
            lines.append(f'{qubits} | {name}{args}')
            lines.extend('  ' + line
                         for line in str(entry.sequence).splitlines())
        return '\n'.join(lines)

    __repr__ = __str__

    def print(self, *, file=None):
        print(str(self), file=file)
