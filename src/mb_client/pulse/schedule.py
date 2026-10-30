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

"""Scheduled pulse timelines and channel-resource scheduling."""

from __future__ import annotations
from dataclasses import dataclass
import copy
from typing import Iterable, Iterator, Mapping, Sequence
import numpy as np
import numpy.typing as npt
from mb_client.common.parameters import Parameter, ParameterExpression, parameters_in
from mb_client.pulse.channels import Channel, ReadoutChannel
from mb_client.pulse.sequence import Pulse, Play, Delay, Barrier, Acquire, ShiftPhase, PulseSequence, _ensure_concrete, _validate_channel, _validated_bias_overrides, _finite_float, _time_array


@dataclass(frozen=True)
class ScheduledPulse:
    channel: Channel
    pulse: Pulse
    start: float | Parameter | ParameterExpression

    def __post_init__(self):
        _validate_channel(self.channel)
        if not isinstance(self.pulse, Pulse):
            raise TypeError("pulse must be a Pulse")
        if isinstance(self.start, (Parameter, ParameterExpression)):
            return
        start = _finite_float(self.start, "start")
        if start < 0:
            raise ValueError("start must be non-negative")
        object.__setattr__(self, "start", start)

    @property
    def end(self) -> float:
        return self.start + self.pulse.duration


@dataclass(frozen=True)
class ScheduledAcquire:
    operation: Acquire
    start: float | Parameter | ParameterExpression

    def __post_init__(self):
        if not isinstance(self.operation, Acquire):
            raise TypeError("operation must be an Acquire")
        if isinstance(self.start, (Parameter, ParameterExpression)):
            return
        start = _finite_float(self.start, "start")
        if start < 0:
            raise ValueError("start must be non-negative")
        object.__setattr__(self, "start", start)

    @property
    def channel(self) -> Channel:
        return self.operation.channel

    @property
    def end(self) -> float:
        return self.start + self.operation.duration


class PulseScheduler:
    """Deterministic as-soon-as-possible scheduler by channel resource."""

    def schedule(self, sequence: PulseSequence) -> PulseSchedule:
        if not isinstance(sequence, PulseSequence):
            raise TypeError("sequence must be a PulseSequence")
        if sequence.template:
            raise ValueError("register template sequences before scheduling")
        _ensure_concrete(sequence)
        available = {}
        frame = {}
        entries = []

        def resources(channel):
            return ((channel, "play"), (channel, "acquire")) if isinstance(
                channel, ReadoutChannel) else ((channel, "play"), )

        for operation in sequence:
            if isinstance(operation, ShiftPhase):
                frame[operation.channel] = frame.get(operation.channel,
                                                     0.0) + operation.phase
            elif isinstance(operation, (Play, Acquire)):
                key = (operation.channel,
                       "acquire" if isinstance(operation, Acquire) else "play")
                start = available.get(key, 0.0)
                if isinstance(operation, Play):
                    pulse = copy.deepcopy(operation.pulse)
                    pulse.phase += frame.get(operation.channel, 0.0)
                    entry = ScheduledPulse(operation.channel, pulse, start)
                else:
                    entry = ScheduledAcquire(operation, start)
                entries.append(entry)
                available[key] = entry.end
            elif isinstance(operation, Delay):
                for key in resources(operation.channel):
                    available[key] = available.get(key,
                                                   0.0) + operation.duration
            elif isinstance(operation, Barrier):
                keys = [
                    key for ch in operation.channels for key in resources(ch)
                ]
                keys = keys or list(available)
                boundary = max((available.get(key, 0.0) for key in keys),
                               default=0.0)
                for key in keys:
                    available[key] = boundary
        return PulseSchedule(entries, bias_overrides=sequence.bias_overrides,
                             metadata=sequence.metadata)


@dataclass(frozen=True, init=False)
class PulseSchedule(Sequence[ScheduledPulse | ScheduledAcquire]):
    """Immutable scheduled pulse timeline."""

    _entries: tuple[ScheduledPulse | ScheduledAcquire, ...]
    metadata: dict[str, object]

    def __init__(self,
                 entries: Iterable[ScheduledPulse | ScheduledAcquire]
                 | None = None,
                 *,
                 bias_overrides: Mapping[int, float] | None = None,
                 metadata: Mapping[str, object] | None = None):
        object.__setattr__(self, "_bias_overrides",
                           _validated_bias_overrides(bias_overrides))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(metadata or {})))
        values = tuple(entries or ())
        if not all(
                isinstance(entry, (ScheduledPulse, ScheduledAcquire))
                for entry in values):
            raise TypeError("entries must contain scheduled pulse operations")
        object.__setattr__(self, "_entries", values)

    def _entry_sequence(self, entry):
        sequence = PulseSequence()
        if isinstance(entry, ScheduledPulse):
            sequence.play(entry.channel, entry.pulse)
        else:
            sequence.append(entry.operation)
        return sequence

    @property
    def parameters(self):
        return frozenset().union(*(self._entry_parameters(entry) for entry in self))

    @property
    def physical_qubits(self):
        return tuple(dict.fromkeys(channel.qubit_index for channel in self.channels))

    def bind_parameters(self, values, *, per_qubit_values=None):
        """Bind pulse/acquisition values and symbolic absolute start times.

        Duration changes do not reschedule later entries or resolve overlaps.
        Static bias overrides remain concrete and are preserved.
        """
        from mb_client.common.parameters import bind_value
        from mb_client.pulse.sequence import transform_sequence
        per_qubit_values = per_qubit_values or {}
        entries = []
        for entry in self:
            bindings = dict(values)
            for parameter, by_qubit in per_qubit_values.items():
                if parameter in self._entry_parameters(entry):
                    qubit = entry.channel.qubit_index
                    if qubit not in by_qubit:
                        raise ValueError(f"missing value for parameter {parameter} on physical qubit {qubit}")
                    bindings[parameter] = by_qubit[qubit]
            operation = transform_sequence(
                self._entry_sequence(entry),
                lambda value: bind_value(value, bindings))[0]
            start = bind_value(entry.start, bindings)
            if isinstance(entry, ScheduledPulse):
                entries.append(ScheduledPulse(entry.channel, operation.pulse, start))
            else:
                entries.append(ScheduledAcquire(operation, start))
        return PulseSchedule(entries, bias_overrides=self.bias_overrides,
                             metadata=self.metadata)

    def _entry_parameters(self, entry):
        from mb_client.pulse.sequence import sequence_parameters
        return frozenset(sequence_parameters(self._entry_sequence(entry))) | parameters_in(entry.start)

    @property
    def bias_overrides(self) -> dict[int, float]:
        """A copy of whole-execution static settings keyed by physical qubit."""
        return dict(self._bias_overrides)

    def __getitem__(self, index):
        return self._entries[index]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ScheduledPulse | ScheduledAcquire]:
        return iter(self._entries)

    @property
    def duration(self) -> float:
        return max((entry.end for entry in self), default=0.0)

    @property
    def channels(self) -> tuple[Channel, ...]:
        return tuple(dict.fromkeys(entry.channel for entry in self))

    def sample(self,
               channel: Channel,
               t: npt.ArrayLike,
               *,
               with_carrier: bool = True) -> np.ndarray:
        _validate_channel(channel)
        if not isinstance(with_carrier, bool):
            raise TypeError("with_carrier must be a bool")
        values = _time_array(t)
        result = np.zeros(values.shape, dtype=complex)
        for entry in self:
            if isinstance(entry, ScheduledPulse) and entry.channel == channel:
                result += entry.pulse.sample(values - entry.start,
                                             with_carrier=with_carrier)
        return result

    def draw(
        self,
        *,
        t: npt.ArrayLike | None = None,
        sample_rate: float = 2e9,
        channels: Iterable[Channel] | None = None,
        group_channel: bool = True,
        with_carrier: bool = False,
        max_pulse_duration: float | None = 100e-9,
        figsize: tuple[float, float] | None = None,
    ):
        from mb_client.visualization.pulse import draw_schedule
        return draw_schedule(self,
                             t=t,
                             sample_rate=sample_rate,
                             channels=channels,
                             group_channel=group_channel,
                             with_carrier=with_carrier,
                             max_pulse_duration=max_pulse_duration,
                             figsize=figsize)

    def __str__(self) -> str:
        from mb_client.visualization.pulse import format_schedule
        return format_schedule(self)

    __repr__ = __str__
