from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pandas as pd
import pytest

from mb_client.circuit import QuantumCircuit
from mb_client.circuit.gates import XGate, SXGate
from mb_client.common import Parameter
from mb_client.compilation import (CalibrationError, CalibrationParams,
                                InstSchedRegistry, Target, lower, schedule)
from mb_client.pulse import Constant, DriveChannel, Pulse, PulseSequence
from mb_client.pulse.sequence import sequence_parameters


def registry(qubits=(0,), amplitude=None):
    result = InstSchedRegistry(list(qubits))
    if amplitude is None:
        amplitude = Parameter('amplitude', group='calibration', default=.1)
    sequence = PulseSequence(template=True)
    sequence.play(DriveChannel(0), Pulse(Constant(20e-9, amplitude)))
    for q in qubits:
        result.add(XGate(), (q,), sequence)
    return result


def params(value, gate=None, qubit=(0,), parameter='amplitude'):
    return CalibrationParams().set(qubit=qubit, parameter=parameter,
                                   value=value, gate=gate)


def test_lookup_merge_replace_and_history():
    target = Target(inst_sched_reg=registry(), calibration_params=params(.2))
    assert target.update_calibration_params(params(.4, 'x')) is None
    assert target.get_calibration_param(0, 'amplitude', gate='x') == .4
    assert target.get_calibration_param(0, 'amplitude', gate='sx') == .2
    assert target.get_calibration_param(0, 'amplitude') == .2
    target.update_calibration_params(params(.3, ''))
    assert target.get_calibration_param(0, 'amplitude') == .3
    frame = target.get_calibration_params().to_dataframe()
    assert len(frame) == 3
    assert frame.gate.iloc[2] is None
    assert target.resolve(XGate(), (0,))[0].pulse.envelope.amplitude == .4
    assert target.resolve(XGate(amplitude=.8), (0,))[0].pulse.envelope.amplitude == .8
    assert target.set_calibration_params(CalibrationParams()) is None
    with pytest.raises(CalibrationError):
        target.get_calibration_param(0, 'amplitude', gate='x')
    assert target.resolve(XGate(), (0,))[0].pulse.envelope.amplitude == .1
    with pytest.raises(ValueError):
        target.get_calibration_param(True, 'amplitude')


def test_getters_and_inputs_are_independent():
    original = registry()
    values = params({'samples': [1]}, parameter='metadata')
    target = Target(inst_sched_reg=original, calibration_params=values)
    original.remove(XGate(), (0,))
    assert target.get_inst_sched_reg().has(XGate(), (0,))
    returned = target.get_inst_sched_reg()
    returned.remove(XGate(), (0,))
    value = target.get_calibration_param(0, 'metadata')
    value['samples'].append(2)
    frame = target.get_calibration_params().to_dataframe()
    frame.loc[0, 'value']['samples'].append(3)
    assert target.get_calibration_param(0, 'metadata') == {'samples': [1]}
    assert target.get_inst_sched_reg().has(XGate(), (0,))
    assert not hasattr(returned, 'load_calibrations')


def test_registry_merge_replacement_and_atomic_failures():
    target = Target(inst_sched_reg=registry(), calibration_params=params(.5))
    patch = registry((0, 2), amplitude=.7)
    assert target.update_inst_sched_reg(patch) is None
    assert target.qubits == (0, 2)
    patch.remove(XGate(), (2,))
    assert target.resolve(XGate(), (2,))[0].pulse.envelope.amplitude == .7
    assert target.set_inst_sched_reg(registry((3,))) is None
    assert target.qubits == (3,)
    assert target.get_calibration_param(0, 'amplitude') == .5
    for method in (target.set_inst_sched_reg, target.update_inst_sched_reg,
                   target.set_calibration_params, target.update_calibration_params):
        with pytest.raises(TypeError):
            method(object())
    invalid = registry((3,))
    invalid.qubits.clear()
    with pytest.raises(ValueError):
        target.set_inst_sched_reg(invalid)
    assert target.qubits == (3,)
    assert target.get_calibration_param(0, 'amplitude') == .5


def test_registry_update_validates_cross_entry_defaults_atomically():
    first = Parameter('amplitude', group='calibration', gate='x', default=.1)
    second = Parameter('amplitude', group='calibration', gate='x', default=.2)
    target = Target(inst_sched_reg=registry(amplitude=first))
    patch = InstSchedRegistry([0])
    sequence = PulseSequence(template=True)
    sequence.play(DriveChannel(0), Pulse(Constant(20e-9, second)))
    patch.add(SXGate(), (0,), sequence)
    with pytest.raises(ValueError, match='Conflicting defaults'):
        target.update_inst_sched_reg(patch)
    assert not target.get_inst_sched_reg().has(SXGate(), (0,))


def test_batch_errors_and_lowering():
    target = Target(inst_sched_reg=registry())
    circuit = QuantumCircuit(1)
    circuit.x(0)
    assert len(lower(circuit, target=target)) == 1
    assert len(schedule(circuit, target=target)) == 1
    assert len(schedule([circuit, circuit], target=target)) == 2
    assert schedule([], target=target) == []
    with pytest.raises(TypeError, match='input 1'):
        schedule([circuit, object()], target=target)
    missing = Target(inst_sched_reg=registry(amplitude=Parameter('amplitude', group='calibration')))
    with pytest.raises(CalibrationError, match='input 0'):
        schedule([circuit], target=missing)
    with pytest.raises(ValueError, match='ordering'):
        schedule([], target=target, ordering='invalid')


def test_batch_uses_one_snapshot_during_concurrent_update(monkeypatch):
    import mb_client.compilation.pulse_lowering as lowering
    target = Target(inst_sched_reg=registry(), calibration_params=params(.2))
    circuit = QuantumCircuit(1)
    circuit.x(0)
    entered, release = Event(), Event()
    original = lowering._lower
    calls = []

    def pause(circuit, snapshot, ordering):
        calls.append(snapshot)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(circuit, snapshot, ordering)

    monkeypatch.setattr(lowering, '_lower', pause)
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(schedule, [circuit, circuit], target=target)
        try:
            assert entered.wait(5)
            target.update_calibration_params(params(.9))
            target.update_inst_sched_reg(registry(amplitude=.6))
        finally:
            release.set()
        batch = future.result(timeout=5)
    assert calls[0] is calls[1]
    assert [item[0].pulse.envelope.amplitude for item in batch] == [.2, .2]
    assert schedule(circuit, target=target)[0].pulse.envelope.amplitude == .6


def test_registry_get_is_symbolic_and_target_binds_expressions():
    from mb_client.circuit.gates import RZGate
    angle = Parameter('angle')
    scale = Parameter('scale', group='calibration', default=2.)
    sequence = PulseSequence(template=True)
    sequence.shift_phase(DriveChannel(0), angle * scale)
    registry = InstSchedRegistry([3])
    registry.add(RZGate(angle), (3,), sequence)
    symbolic = registry.get(RZGate(.5), (3,))
    assert set(sequence_parameters(symbolic)) == {angle, scale}
    assert symbolic[0].channel == DriveChannel(3)
    symbolic.play(DriveChannel(3), Pulse(Constant(1e-9, .1)))
    assert len(registry.get('rz', (3,))) == 1
    target = Target(inst_sched_reg=registry)
    resolved = target.resolve(RZGate(.5), (3,))
    assert not sequence_parameters(resolved)
    assert resolved[0].phase == 1.
    assert set(sequence_parameters(registry.get('rz', (3,)))) == {angle, scale}


def test_only_target_validates_resolution_arguments_and_overrides():
    from mb_client.circuit import Instruction
    registrations = InstSchedRegistry([0])
    sequence = PulseSequence(template=True)
    sequence.play(DriveChannel(0), Pulse(Constant(20e-9, .1)))
    registrations.add(Instruction('fixed', 1, params=[2.]), (0,), sequence)
    target = Target(inst_sched_reg=registrations)
    for params_, message in [([], 'argument count mismatch'), ([3.], 'fixed gate argument mismatch')]:
        gate = Instruction('fixed', 1, params=params_)
        assert len(registrations.get(gate, (0,))) == 1
        with pytest.raises(ValueError, match=message):
            target.resolve(gate, (0,))
    registrations = registry()
    gate = XGate(unknown=.2)
    assert len(registrations.get(gate, (0,))) == 1
    with pytest.raises(ValueError, match='unknown calibration overrides'):
        Target(inst_sched_reg=registrations).resolve(gate, (0,))
