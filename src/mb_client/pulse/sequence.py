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

"""Pulse values, sequence operations, and symbolic transformations."""

from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence, TypeAlias
from dataclasses import dataclass, replace
import copy
import math
from numbers import Real
import numpy as np
import numpy.typing as npt
from mb_client.common.parameters import Parameter, ParameterExpression
from mb_client.pulse.envelopes import Envelope, Idle
from mb_client.pulse.channels import Channel, DriveChannel, FluxChannel, ReadoutChannel, _Channel

if TYPE_CHECKING:
    from .schedule import PulseScheduler, PulseSchedule

SYMBOLIC: TypeAlias = Parameter | ParameterExpression


def _template_value(value, name: str):
    if isinstance(value, SYMBOLIC):
        return value
    return _finite_float(value, name)


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _time_array(t: npt.ArrayLike) -> np.ndarray:
    try:
        result = np.asarray(t, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("time values must be real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("time values must be finite")
    return result


def _duration_mask(t: np.ndarray, duration: float) -> np.ndarray:
    tolerance = max(1e-15, abs(duration) * 1e-12)
    return (t >= -tolerance) & (t <= duration + tolerance)


class Pulse:
    """An envelope scaled and modulated by a carrier."""

    def __init__(self,
                 envelope: Envelope,
                 *,
                 frequency: float = 0.0,
                 phase: float = 0.0,
                 offset: float = 0.0):
        if not isinstance(envelope, Envelope):
            raise TypeError("envelope must be an Envelope")
        self.envelope = envelope
        self.frequency = _template_value(frequency, "frequency")
        self.phase = _template_value(phase, "phase")
        self.offset = _template_value(offset, "offset")

    @property
    def duration(self) -> float:
        return self.envelope.duration

    def sample(self,
               t: npt.ArrayLike,
               *,
               with_carrier: bool = True) -> np.ndarray:
        if not isinstance(with_carrier, bool):
            raise TypeError("with_carrier must be a bool")
        values = _time_array(t)
        envelope = self.envelope.sample(values)
        carrier_frequency = self.frequency if with_carrier else 0.0
        carrier = np.exp(-2j * np.pi * carrier_frequency * values +
                         1j * self.phase)
        result = envelope * carrier + self.offset
        if not np.all(np.isfinite(result)):
            raise ValueError("pulse sampling returned non-finite values")
        return np.asarray(result, dtype=complex)


@dataclass(frozen=True)
class Play:
    channel: Channel
    pulse: Pulse

    def __post_init__(self):
        _validate_channel(self.channel)
        if not isinstance(self.pulse, Pulse):
            raise TypeError("pulse must be a Pulse")


@dataclass(frozen=True)
class Delay:
    channel: Channel
    duration: float

    def __post_init__(self):
        _validate_channel(self.channel)
        duration = _template_value(self.duration, "duration")
        if not isinstance(duration, SYMBOLIC) and duration < 0:
            raise ValueError("delay duration must be non-negative")
        object.__setattr__(self, "duration", duration)


@dataclass(frozen=True)
class Barrier:
    """Synchronize channel resources without emitting idle pulses."""

    channels: tuple[Channel, ...] = ()

    def __post_init__(self):
        channels = tuple(self.channels)
        for channel in channels:
            _validate_channel(channel)
        if len(set(channels)) != len(channels):
            raise ValueError("barrier channels must be unique")
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class Acquire:
    """Acquisition with an optional destination classical-bit index."""

    channel: Channel
    duration: float
    digitizer_delay: float = 0.0
    demod_freq: float = 0.0
    data_form: str = "DEMOD"
    name: str | None = None
    clbit: int | None = None

    def __post_init__(self):
        _validate_channel(self.channel)
        if not isinstance(self.channel, ReadoutChannel):
            raise TypeError("acquisition requires ReadoutChannel")
        if self.clbit is not None:
            if type(self.clbit) is not int:
                raise TypeError("clbit must be an integer or None")
            if self.clbit < 0:
                raise ValueError("clbit must be non-negative")
        for field in ("duration", "digitizer_delay", "demod_freq"):
            item = _template_value(getattr(self, field), field)
            if (field != "demod_freq" and not isinstance(item, SYMBOLIC)
                    and item < 0):
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, item)
        if not isinstance(self.data_form, str) or not self.data_form:
            raise ValueError("data_form must be non-empty")
        if self.name is not None and (not isinstance(self.name, str)
                                      or not self.name):
            raise ValueError("acquisition name must be a non-empty string")


@dataclass(frozen=True)
class ShiftPhase:
    channel: Channel
    phase: object

    def __post_init__(self):
        _validate_channel(self.channel)
        object.__setattr__(self, "phase", _template_value(self.phase, "phase"))


def _validate_channel(channel: Channel) -> None:
    if not isinstance(channel, (DriveChannel, FluxChannel, ReadoutChannel)):
        raise TypeError(
            "channel must be DriveChannel, FluxChannel, or ReadoutChannel")


def _validated_bias_overrides(values):
    result = {}
    for qubit, value in (values or {}).items():
        if type(qubit) is not int or qubit < 0:
            raise ValueError(
                "bias overrides require non-negative physical qubit IDs")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("bias must be a finite real number")
        if not math.isfinite(value):
            raise ValueError("bias must be a finite real number")
        result[qubit] = float(value)
    return result


Operation: TypeAlias = Play | Delay | Barrier | Acquire | ShiftPhase


class PulseSequence(Sequence[Operation]):
    """Ordered pulse operations with independently copied user metadata."""

    def __init__(self,
                 operations: Iterable[Operation] | None = None,
                 *,
                 template: bool = False,
                 bias_overrides: Mapping[int, float] | None = None,
                 metadata: Mapping[str, object] | None = None):
        if type(template) is not bool:
            raise TypeError("template must be a bool")
        self._template = template
        self.metadata: dict[str, object] = copy.deepcopy(dict(metadata or {}))
        if template and bias_overrides:
            raise ValueError(
                "static bias belongs to an execution, not a pulse template")
        self._bias_overrides = _validated_bias_overrides(bias_overrides)
        self._operations: list[Operation] = []
        for operation in operations or ():
            self.append(operation)

    def __enter__(self) -> PulseSequence:
        """Return this sequence for explicit construction in a with block."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Keep appended operations and propagate any exception."""
        return False

    @property
    def bias_overrides(self) -> dict[int, float]:
        """Static settings by physical qubit, applied before the whole execution.

        Returns a copy; these settings have no pulse duration or timeline position.
        """
        return dict(self._bias_overrides)

    @property
    def template(self):
        return self._template

    def __getitem__(self, index):
        return self._operations[index]

    def __len__(self) -> int:
        return len(self._operations)

    def append(self, item: Operation) -> Operation:
        if (isinstance(item, tuple) and len(item) == 2
                and isinstance(item[0], _Channel)):
            item = Play(item[0], item[1])
        if not isinstance(item, (Play, Delay, Barrier, Acquire, ShiftPhase)):
            raise TypeError(
                "item must be a (channel, pulse), Play, Delay, Barrier, or Acquire"
            )
        self._operations.append(item)
        return item

    def play(self, channel: Channel, pulse: Pulse) -> Play:
        operation = Play(channel, pulse)
        self._operations.append(operation)
        return operation

    def delay(self, channel: Channel, duration: float) -> Delay:
        operation = Delay(channel, duration)
        self._operations.append(operation)
        return operation

    def shift_phase(self, channel: Channel, phase) -> ShiftPhase:
        return self.append(ShiftPhase(channel, phase))

    def barrier(self, *channels: Channel) -> Barrier:
        operation = Barrier(tuple(channels))
        self._operations.append(operation)
        return operation

    def acquire(self, channel: Channel, duration: float, **options) -> Acquire:
        operation = Acquire(channel, duration, **options)
        self._operations.append(operation)
        return operation

    def schedule(self,
                 scheduler: PulseScheduler | None = None) -> PulseSchedule:
        from .schedule import PulseScheduler
        return (scheduler or PulseScheduler()).schedule(self)

    def __str__(self) -> str:
        from mb_client.visualization.pulse import format_sequence
        return format_sequence(self)

    __repr__ = __str__


def _ensure_concrete(sequence: PulseSequence) -> None:

    def check(value, field):
        if isinstance(value, SYMBOLIC):
            raise ValueError(f"unresolved parameter {value!s} in {field}")

    for operation in sequence:
        if isinstance(operation, ShiftPhase):
            check(operation.phase, "phase shift")
        elif isinstance(operation, Delay):
            check(operation.duration, "delay duration")
        elif isinstance(operation, Play):
            pulse = operation.pulse
            check(pulse.envelope.duration, "envelope duration")
            check(pulse.envelope.amplitude, "envelope amplitude")
            check(getattr(pulse.envelope, "beta", None), "envelope beta")
            check(getattr(pulse.envelope, "sigma", None), "envelope sigma")
            check(getattr(pulse.envelope, "detuning", None),
                  "envelope detuning")
            check(getattr(pulse.envelope, "omega_0", None), "envelope omega_0")
            for field in ("frequency", "phase", "offset"):
                check(getattr(pulse, field), f"pulse {field}")
        elif isinstance(operation, Acquire):
            for field in ("duration", "digitizer_delay", "demod_freq"):
                check(getattr(operation, field), f"acquire {field}")


def discover_object_data_member(obj):
    result = []
    for name in vars(obj):
        if 'frozen' in name:
            continue
        if name.startswith('_'):
            result.append(name[1:])
    return result


def transform_sequence(sequence,
                       transform_value=lambda x: x,
                       transform_channel=lambda x: x,
                       *,
                       clbit: int | None = None):
    """Rebuild operations so resolved numeric values pass constructor validation.

    A non-None clbit overrides every acquisition's classical destination;
    None preserves each acquisition's existing destination.
    """
    result = PulseSequence(metadata=sequence.metadata)
    for op in sequence:
        if isinstance(op, Play):
            pulse = op.pulse
            env = pulse.envelope
            kwargs = {
                name: transform_value(getattr(env, name))
                for name in discover_object_data_member(env)
                if hasattr(env, name)
            }
            if isinstance(env, Idle):
                kwargs.pop('amplitude', None)
            envelope = type(env)(**kwargs)
            result.play(
                transform_channel(op.channel),
                Pulse(
                    envelope, **{
                        name: transform_value(getattr(pulse, name))
                        for name in ('frequency', 'phase', 'offset')
                    }))
        elif isinstance(op, Barrier):
            result.append(
                replace(op,
                        channels=tuple(
                            transform_channel(ch) for ch in op.channels)))
        elif isinstance(op, Delay):
            result.delay(transform_channel(op.channel),
                         transform_value(op.duration))
        elif isinstance(op, ShiftPhase):
            result.shift_phase(transform_channel(op.channel),
                               transform_value(op.phase))
        elif isinstance(op, Acquire):
            result.acquire(transform_channel(op.channel),
                           transform_value(op.duration),
                           digitizer_delay=transform_value(op.digitizer_delay),
                           demod_freq=transform_value(op.demod_freq),
                           data_form=op.data_form,
                           name=op.name,
                           clbit=op.clbit if clbit is None else clbit)
        else:
            raise TypeError(
                f'Unsupported sequence operation {type(op).__name__}')
    return result


def sequence_parameters(sequence):
    found = []

    def visit(value):
        if isinstance(value, Parameter):
            found.append(value)
        elif isinstance(value, ParameterExpression):
            for operand in value.operands:
                visit(operand)
        return value

    transform_sequence(sequence, transform_value=visit)
    return found
