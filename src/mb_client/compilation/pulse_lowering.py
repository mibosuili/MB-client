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

"""compilation: pulse lowering APIs."""

from __future__ import annotations
from dataclasses import dataclass
import copy
import math
from mb_client.compilation.calibration import CalibrationError
from mb_client.circuit import Barrier, Delay, Drive, IGate, CircuitInstruction, Instruction, QuantumCircuit
from mb_client.compilation.target import Target
from collections.abc import Sequence
from typing import Literal, overload
from mb_client.pulse import Barrier as PulseBarrier, Channel, DriveChannel, FluxChannel, ReadoutChannel, PulseSequence, PulseScheduler, PulseSchedule, Pulse, Idle
from mb_client.pulse.sequence import _ensure_concrete, Play, Acquire, Delay as PulseDelay
from mb_client.pulse.schedule import ScheduledAcquire
from mb_client.common.parameters import parameters_in
from mb_client.pulse.block import PulseBlock


@dataclass(frozen=True)
class LoweredInstruction:
    """A resolved sequence and the metadata needed to preserve gate ordering."""

    qubits: tuple[int, ...]
    sequence: PulseSequence
    is_measurement: bool = False


@dataclass(frozen=True)
class LoweredBarrier:
    """An explicit barrier over mapped physical operands."""

    qubits: tuple[int, ...]


@dataclass(frozen=True)
class LoweredCircuit:
    """Retain the root mapping, including qubits without instructions."""

    physical_qubits: tuple[int, ...]
    instructions: tuple[LoweredInstruction | LoweredBarrier, ...]
    bias_overrides: dict[int, float]


class LowerInstructionsPass:
    """Validate visited instructions and resolve them into pulse records.

    Registrations take precedence over delays and definitions. A resolution error
    propagates without trying a definition; unvisited definitions are not checked.
    Traversal state belongs to each run, never to the pass instance.
    """

    def __init__(self, target: Target):
        self._target = target
        self._registry = target.get_inst_sched_reg()

    def run(self, circuit: QuantumCircuit) -> LoweredCircuit:
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be a QuantumCircuit")
        mapping = tuple(circuit.physical_qubits)
        if (len(mapping) != circuit.num_qubits or any(
                type(q) is not int or q not in self._target.qubits
                for q in mapping) or len(set(mapping)) != len(mapping)):
            raise ValueError(
                f"invalid circuit physical-qubit mapping {mapping}")
        lowered: list[LoweredInstruction | LoweredBarrier] = []
        self._visit(circuit, mapping, tuple(range(circuit.num_clbits)),
                    circuit.name, set(), lowered)
        overrides = circuit.bias
        if parameters_in(overrides):
            raise ValueError("unbound parameters in circuit bias")
        bias = {}
        calibrations = self._target.get_calibration_params()
        for logical, physical in enumerate(mapping):
            if logical in overrides:
                bias[physical] = overrides[logical]
                continue
            try:
                value = calibrations.get((physical,), "dc_bias")
            except CalibrationError:
                continue
            if value is not None:
                bias[physical] = value
        return LoweredCircuit(mapping, tuple(lowered), bias)

    def _visit(self, body: QuantumCircuit, physical: tuple[int, ...],
               classical: tuple[int, ...], path: str, active: set[int],
               lowered: list[LoweredInstruction | LoweredBarrier]) -> None:
        if id(body) in active:
            raise ValueError(
                f"{path}: cyclic gate decomposition on {physical}")
        active = active | {id(body)}
        for index, item in enumerate(body.data):
            operation = item.operation
            location = f"{path}/{index}:{getattr(operation, 'name', '?')}"
            qubits, cbits = self._map_operands(item, body, physical, classical,
                                               location)
            context = f"{location} on {qubits}"
            try:
                self._validate_operation(operation, qubits, cbits)
                self._lower_operation(operation, qubits, cbits, physical,
                                      location, active, lowered)
            except (ValueError, TypeError, KeyError,
                    NotImplementedError) as exc:
                raise type(exc)(f"{context}: {exc}") from exc

    @staticmethod
    def _map_operands(
        item: CircuitInstruction,
        body: QuantumCircuit,
        physical: tuple[int, ...],
        classical: tuple[int, ...],
        location: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        operation = item.operation
        if not isinstance(operation, Instruction):
            raise TypeError(f"{location}: operation must be Instruction")
        if (len(item.qubits) != operation.num_qubits
                or len(item.clbits) != operation.num_clbits):
            raise ValueError(f"{location}: instruction arity mismatch")
        for bits, count, label in ((item.qubits, body.num_qubits, 'qubit'),
                                   (item.clbits, body.num_clbits, 'clbit')):
            if (any(
                    type(bit.index) is not int or not 0 <= bit.index < count
                    for bit in bits) or len(set(bits)) != len(bits)):
                raise ValueError(f"{location}: invalid {label} operands")
        return (tuple(physical[q.index] for q in item.qubits),
                tuple(classical[c.index] for c in item.clbits))

    @staticmethod
    def _validate_operation(operation: Instruction, qubits: tuple[int, ...],
                            cbits: tuple[int, ...]) -> None:
        unresolved = set(parameters_in(operation.params))
        for attribute in ('calibration', 'parameters'):
            unresolved.update(parameters_in(getattr(operation, attribute, {})))
        if unresolved:
            raise ValueError(
                f"unbound parameters: {sorted(map(str, unresolved))}")
        if isinstance(operation, Drive):
            raise NotImplementedError(
                "device-dependent operation; use an explicitly registered custom instruction"
            )
        measurement = operation.name == 'measure' and not isinstance(
            operation, PulseBlock)
        if cbits and not measurement:
            raise NotImplementedError(
                "classical operands are supported only for measurement")
        if measurement:
            if len(qubits) != 1 or len(cbits) != 1:
                raise ValueError(
                    "measurement requires one qubit and one clbit")
            unsupported = {'z_offset', 'final_align'} & set(
                getattr(operation, 'calibration', {}))
            if unsupported:
                raise NotImplementedError(
                    f"legacy measurement options: {sorted(unsupported)}")

    def _lower_operation(
        self,
        operation: Instruction,
        qubits: tuple[int, ...],
        cbits: tuple[int, ...],
        physical: tuple[int, ...],
        location: str,
        active: set[int],
        lowered: list[LoweredInstruction | LoweredBarrier],
    ) -> None:
        if isinstance(operation, Barrier):
            lowered.append(LoweredBarrier(qubits or physical))
            return
        if isinstance(operation, PulseBlock):
            sequence = operation.lower(qubits)
        elif self._registry.has(operation, qubits):
            sequence = self._target.resolve(operation, qubits, cbits=cbits)
        elif isinstance(operation, IGate):
            sequence = self._lower_idle(operation, qubits)
        elif isinstance(operation, Delay):
            sequence = self._lower_delay(operation, qubits)
        elif operation.definition is not None:
            self._lower_definition(operation.definition, qubits, cbits,
                                   location, active, lowered)
            return
        else:
            raise KeyError(
                f"no PulseSequence or definition for {operation.name!r}")
        lowered.append(LoweredInstruction(
            qubits, sequence, operation.name == 'measure' and bool(cbits)))

    @staticmethod
    def _lower_idle(
        operation: IGate,
        qubits: tuple[int, ...],
    ) -> PulseSequence:
        if len(operation.params) != 1:
            raise ValueError("idle gate requires an explicit duration")

        duration = operation.params[0]
        channels = _boundary_channels(qubits)

        sequence = PulseSequence()
        sequence.barrier(*channels)
        for channel in channels:
            sequence.delay(channel, duration)
        sequence.barrier(*channels)
        return sequence

    @staticmethod
    def _lower_delay(operation: Delay, qubits: tuple[int,
                                                     ...]) -> PulseSequence:
        duration, selector = operation.params
        channels = {'drive': DriveChannel, 'flux': FluxChannel, 'readout': ReadoutChannel}
        if selector not in channels:
            raise ValueError(f"unsupported delay channel {selector!r}")
        sequence = PulseSequence()
        sequence.delay(channels[selector](qubits[0]), duration)
        return sequence

    def _lower_definition(
        self,
        definition: QuantumCircuit,
        qubits: tuple[int, ...],
        cbits: tuple[int, ...],
        location: str,
        active: set[int],
        lowered: list[LoweredInstruction | LoweredBarrier],
    ) -> None:
        if definition.bias:
            raise ValueError("gate definitions cannot carry static bias")
        if (definition.num_qubits != len(qubits)
                or definition.num_clbits != len(cbits)):
            raise ValueError("definition arity mismatch")
        self._visit(definition, qubits, cbits, location, active, lowered)


def _boundary_channels(qubits: tuple[int, ...]) -> tuple[Channel, ...]:
    """Preserve operand order and drive/flux/readout order within each operand."""
    return tuple(
        channel(q) for q in qubits
        for channel in (DriveChannel, FluxChannel, ReadoutChannel))


class AlignFinalMeasurementsPass:
    """Align terminal acquisitions without crossing explicit circuit barriers.

    Independent operations may precede deferred terminal measurements. A later
    operation on the same qubit prevents a measurement from being terminal.
    Calibration-relative pulse/acquisition timing is preserved by padding the
    whole measurement sequence.
    """

    def run(self, circuit: LoweredCircuit) -> LoweredCircuit:
        instructions = list(circuit.instructions)
        end = len(instructions)
        while end and isinstance(instructions[end - 1], LoweredBarrier):
            end -= 1
        blocked = set()
        selected = []
        for index in range(end - 1, -1, -1):
            instruction = instructions[index]
            if isinstance(instruction, LoweredBarrier):
                break
            resources = set(instruction.qubits)
            for item in instruction.sequence:
                channels = (item.channels if isinstance(item, PulseBarrier)
                            else (item.channel,))
                # An unscoped pulse barrier may synchronize other operands.
                if isinstance(item, PulseBarrier) and not channels:
                    resources.update(circuit.physical_qubits)
                resources.update(channel.qubit_index for channel in channels)
            if instruction.is_measurement and not resources & blocked:
                selected.append(index)
            blocked.update(resources)
        if len(selected) < 2:
            return circuit
        selected.reverse()
        measurements = [instructions[index] for index in selected]
        starts = []
        for instruction in measurements:
            acquisitions = [entry for entry in instruction.sequence.schedule()
                            if isinstance(entry, ScheduledAcquire)]
            if len(acquisitions) != 1:
                raise ValueError("final measurement alignment requires one acquisition per measurement")
            starts.append(acquisitions[0].start)
        latest = max(starts)
        qubits = tuple(dict.fromkeys(q for instruction in measurements
                                    for q in instruction.qubits))
        selected_set = set(selected)
        result = [item for index, item in enumerate(instructions[:end])
                  if index not in selected_set]
        result.append(LoweredBarrier(qubits))
        for instruction, start in zip(measurements, starts):
            sequence = PulseSequence(bias_overrides=instruction.sequence.bias_overrides)
            if latest > start:
                for channel in _boundary_channels(instruction.qubits):
                    sequence.delay(channel, latest - start)
            for item in instruction.sequence:
                sequence.append(item)
            result.append(LoweredInstruction(instruction.qubits, sequence, True))
        result.extend(instructions[end:])
        return LoweredCircuit(circuit.physical_qubits, tuple(result),
                              circuit.bias_overrides)


class PadMeasurementDrivePass:
    """Keep each measured qubit's XY channel idle through readout completion.

    Run after final-measurement alignment so any alignment delays are included.
    Only channels with preceding or in-template drive pulses are padded.
    Delays and barriers alone do not activate a drive channel.
    """

    def run(self, circuit: LoweredCircuit) -> LoweredCircuit:
        instructions = []
        active_drives = set()
        for instruction in circuit.instructions:
            if not isinstance(instruction, LoweredBarrier):
                active_drives.update(item.channel for item in instruction.sequence
                                     if isinstance(item, Play)
                                     and isinstance(item.channel, DriveChannel))
            if isinstance(instruction, LoweredBarrier) or not instruction.is_measurement:
                instructions.append(instruction)
                continue
            sequence = copy.deepcopy(instruction.sequence)
            scheduled = sequence.schedule()
            for qubit in instruction.qubits:
                if DriveChannel(qubit) not in active_drives:
                    continue
                end = max((entry.end for entry in scheduled
                           if entry.channel == ReadoutChannel(qubit)), default=0.0)
                # A discarded probe observes the drive cursor, including delays
                # and barriers that do not appear in a scheduled pulse list.
                probe = copy.deepcopy(sequence)
                probe.play(DriveChannel(qubit), Pulse(Idle(1.0)))
                start = probe.schedule()[-1].start
                if end > start:
                    sequence.play(DriveChannel(qubit), Pulse(Idle(end - start)))
            instructions.append(LoweredInstruction(
                instruction.qubits, sequence, instruction.is_measurement))
        return LoweredCircuit(circuit.physical_qubits, tuple(instructions),
                              circuit.bias_overrides)


class AssemblePulseSequencePass:
    """Assemble channel timelines or impose gate boundaries on each operand.

    Explicit barriers are preserved in either policy.
    """

    def __init__(self, ordering: str = "channel"):
        if ordering not in ("channel", "gate"):
            raise ValueError("ordering must be 'channel' or 'gate'")
        self.ordering = ordering

    def run(self, circuit: LoweredCircuit) -> PulseSequence:
        result = PulseSequence(bias_overrides=circuit.bias_overrides)
        for instruction in circuit.instructions:
            boundary = _boundary_channels(instruction.qubits)
            if isinstance(instruction, LoweredBarrier):
                if boundary:
                    result.append(PulseBarrier(boundary))
                continue
            synchronize = self.ordering == "gate"
            if synchronize and boundary:
                result.append(PulseBarrier(boundary))
            elif instruction.is_measurement:
                # Readout starts after preceding XY activity on its own qubit.
                # Keep Z and other qubits independent in channel ordering.
                for qubit in instruction.qubits:
                    result.append(PulseBarrier((DriveChannel(qubit),
                                                ReadoutChannel(qubit))))
            for item in instruction.sequence:
                result.append(item)
            if synchronize and boundary:
                result.append(PulseBarrier(boundary))
        return result


class MaterializeBarrierPaddingPass:
    """Emit barrier-wait idles on already-used drive/flux channels, preserving timing."""
    def run(self, sequence: PulseSequence) -> PulseSequence:
        result = PulseSequence(bias_overrides=sequence.bias_overrides)
        result.metadata = copy.deepcopy(sequence.metadata)
        available, active = {}, set()
        def resources(channel):
            return ((channel, 'play'), (channel, 'acquire')) if isinstance(
                channel, ReadoutChannel) else ((channel, 'play'),)
        for operation in sequence:
            if isinstance(operation, Play):
                key = (operation.channel, 'play')
                available[key] = available.get(key, 0.) + operation.pulse.duration
                active.add(operation.channel)
            elif isinstance(operation, Acquire):
                key = (operation.channel, 'acquire')
                available[key] = available.get(key, 0.) + operation.duration
            elif isinstance(operation, PulseDelay):
                for key in resources(operation.channel):
                    available[key] = available.get(key, 0.) + operation.duration
            elif isinstance(operation, PulseBarrier):
                keys = [key for channel in operation.channels for key in resources(channel)]
                keys = keys or list(available)
                end = max((available.get(key, 0.) for key in keys), default=0.)
                for key in dict.fromkeys(keys):
                    channel, resource = key
                    start = available.get(key, 0.)
                    if (resource == 'play' and channel in active
                            and isinstance(channel, (DriveChannel, FluxChannel))
                            and end > start
                            and not math.isclose(end, start, rel_tol=1e-12, abs_tol=1e-18)):
                        result.play(channel, Pulse(Idle(end-start)))
                    available[key] = end
            result.append(copy.deepcopy(operation))
        return result


class ValidateConcreteSequencePass:
    """Check the final sequence for unresolved values and return it unchanged."""

    def run(self, sequence: PulseSequence) -> PulseSequence:
        _ensure_concrete(sequence)
        return sequence


def _lower(circuit: QuantumCircuit, target: Target, ordering: str) -> PulseSequence:
    lowered = LowerInstructionsPass(target).run(circuit)
    lowered = AlignFinalMeasurementsPass().run(lowered)
    #lowered = PadMeasurementDrivePass().run(lowered)
    lowered = AssemblePulseSequencePass(ordering).run(lowered)
    #lowered = MaterializeBarrierPaddingPass().run(lowered)
    lowered = ValidateConcreteSequencePass().run(lowered)
    lowered.metadata = copy.deepcopy(circuit.metadata)
    return lowered

def _snapshot(target: Target) -> Target:
    if not isinstance(target, Target):
        raise TypeError("target must be Target")
    return target._snapshot()


def lower(circuit: QuantumCircuit, *, target: Target,
          ordering: Literal["channel", "gate"] = "channel") -> PulseSequence:
    """Lower a bound circuit using one isolated snapshot of the target.

    Circuit bias overrides calibrated dc_bias. No basis translation or hardware
    conversion is performed. Gate ordering synchronizes instruction operands;
    channel ordering synchronizes explicit barriers and final measurements.
    """
    return _lower(circuit, _snapshot(target), ordering)


@overload
def schedule(circuits: QuantumCircuit, *, target: Target,
             ordering: Literal["channel", "gate"] = "channel") -> PulseSchedule: ...


@overload
def schedule(circuits: Sequence[QuantumCircuit], *, target: Target,
             ordering: Literal["channel", "gate"] = "channel") -> list[PulseSchedule]: ...


def schedule(circuits: QuantumCircuit | Sequence[QuantumCircuit], *, target: Target,
             ordering: Literal["channel", "gate"] = "channel") -> PulseSchedule | list[PulseSchedule]:
    """Schedule one bound circuit or an ordered batch against one snapshot.

    Empty batches return []. Batch failures include the input index and return
    no partial results. Updates to the original target affect future calls only.
    """
    if ordering not in ("channel", "gate"):
        raise ValueError("ordering must be 'channel' or 'gate'")
    snapshot = _snapshot(target)
    scheduler = PulseScheduler()
    if isinstance(circuits, QuantumCircuit):
        return scheduler.schedule(_lower(circuits, snapshot, ordering))
    if not isinstance(circuits, Sequence) or isinstance(circuits, (str, bytes)):
        raise TypeError("circuits must be a QuantumCircuit or a sequence of circuits")
    results = []
    for index, circuit in enumerate(circuits):
        try:
            results.append(scheduler.schedule(_lower(circuit, snapshot, ordering)))
        except (ValueError, TypeError, LookupError, NotImplementedError) as exc:
            raise type(exc)(f"input {index}: {exc}") from exc
    return results
