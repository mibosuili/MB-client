import numpy as np
import pandas as pd
import pytest

from mb_client.common import Parameter
from mb_client.circuit import QuantumCircuit
from mb_client.circuit.gates import RZGate, SXGate, XGate
from mb_client.compilation import (CalibrationError, CalibrationParams,
                                InstSchedRegistry, Target, lower, schedule as schedule_circuits,
                                default_inst_sched_registry, translate_basis)
from mb_client.pulse import (Constant, DriveChannel, Pulse, PulseSchedule,
                          PulseSequence, ReadoutChannel, ScheduledAcquire)


def calibrated_target(qubits=(2, )):
    rows = []
    for qubit in qubits:
        for parameter, value in [('amplitude', .2), ('duration', 20e-9),
                                 ('detuning', 0.), ('beta', 0.),
                                 ('qubit_frequency', 5e9),
                                 ('cavity_frequency', 6e9), ('post_pad', 5e-9),
                                 ('dc_bias', .1)]:
            rows.append(
                dict(qubit=(qubit,), parameter=parameter, value=value))
    registrations = default_inst_sched_registry(list(qubits))
    angle = Parameter('angle')
    frame = PulseSequence(template=True)
    frame.shift_phase(DriveChannel(0), angle)
    for q in qubits:
        registrations.set(RZGate(angle), (q, ), frame)
    return Target(inst_sched_reg=registrations, calibration_params=CalibrationParams(pd.DataFrame(rows)))


def test_translate_lower_schedule_and_snapshot():
    registrations = calibrated_target()
    target = registrations
    circuit = QuantumCircuit(1, 1, bias={0: .3})
    circuit.physical_qubits = (2, )
    circuit.h(0)
    circuit.measure(0, 0)
    translated = translate_basis(circuit)
    assert {i.operation.name for i in translated} <= {'rz', 'sx', 'measure'}
    assert translated.bias == {0: .3}
    assert translated.physical_qubits == (2, )
    sequence = lower(translated, target=target, ordering='gate')
    schedule = schedule_circuits(translated, target=target, ordering='gate')
    assert isinstance(sequence, PulseSequence)
    assert isinstance(schedule, PulseSchedule)
    assert schedule.bias_overrides == {2: .3}
    assert any(isinstance(entry, ScheduledAcquire) for entry in schedule)
    assert schedule[-1].start >= 20e-9
    snapshot = target.get_inst_sched_reg()
    snapshot.remove(SXGate(), (2, ))
    assert target.get_inst_sched_reg().has(SXGate(), (2, ))


def test_pulse_block_binds_and_maps_without_registry():
    duration = Parameter('duration')
    sequence = PulseSequence(template=True)
    sequence.play(DriveChannel(0), Pulse(Constant(duration, .4)))
    circuit = QuantumCircuit(1)
    circuit.physical_qubits = (5, )
    circuit.pulse_block(sequence, [0])
    target = Target(inst_sched_reg=InstSchedRegistry([5]))
    with pytest.raises(ValueError, match='unbound'):
        lower(circuit, target=target)
    bound = circuit.bind_parameters({duration: 20e-9})
    schedule = schedule_circuits(bound, target=target)
    assert schedule[0].channel == DriveChannel(5)
    np.testing.assert_allclose(schedule.sample(DriveChannel(5), [0, 10e-9]),
                               [.4, .4])
    assert circuit.parameters == {duration}
    assert 'PulseSequence' in str(lower(bound, target=target))


def test_sequence_scheduling_preserves_phase_barrier_and_bias():
    sequence = PulseSequence(bias_overrides={3: .25})
    sequence.shift_phase(DriveChannel(3), np.pi / 2)
    sequence.play(DriveChannel(3), Pulse(Constant(20e-9, .5)))
    sequence.barrier(DriveChannel(3), ReadoutChannel(3))
    sequence.acquire(ReadoutChannel(3), 40e-9)
    schedule = sequence.schedule()
    assert schedule[1].start == 20e-9
    assert schedule.bias_overrides == {3: .25}
    np.testing.assert_allclose(schedule.sample(DriveChannel(3), [10e-9]),
                               [.5j],
                               atol=1e-15)


def test_calibration_overrides_defaults_and_missing_values():
    table = CalibrationParams(
        pd.DataFrame([
            dict(qubit=(0,), parameter='amplitude', value=.2),
            dict(qubit=(0,), parameter='amplitude', value=.4, gate='x'),
        ]))
    assert table.get((0,), 'amplitude', gate='x') == .4
    assert table.get((0,), 'amplitude', gate='sx') == .2
    with pytest.raises(CalibrationError):
        table.get((0,), 'missing')
    registrations = InstSchedRegistry([0])
    amplitude = Parameter('amplitude', group='calibration', default=.1)
    sequence = PulseSequence(template=True)
    sequence.play(DriveChannel(0), Pulse(Constant(10e-9, amplitude)))
    registrations.add(XGate(), (0, ), sequence)
    target = Target(inst_sched_reg=registrations)
    assert target.resolve(XGate(), (0, ))[0].pulse.envelope.amplitude == .1
    target.set_calibration_params(table)
    assert target.resolve(XGate(amplitude=.8), (0, ))[0].pulse.envelope.amplitude == .8
