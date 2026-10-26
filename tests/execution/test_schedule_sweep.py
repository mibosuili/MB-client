import numpy as np
import pytest
from mb_client.common import Parameter
from mb_client.compilation import CalibrationParams, InstSchedRegistry, Target
from mb_client.execution import Sweep, SweepAxis
from mb_client.pulse import Constant, DriveChannel, ReadoutChannel, Pulse
from mb_client.pulse.sequence import Acquire
from mb_client.pulse.schedule import PulseSchedule, ScheduledPulse, ScheduledAcquire


def test_schedule_grid_preserves_starts_acquisitions_and_bias():
    amp, duration = Parameter('amp'), Parameter('duration')
    program = PulseSchedule([
        ScheduledPulse(DriveChannel(0), Pulse(Constant(duration, 2 * amp)), 10e-9),
        ScheduledAcquire(Acquire(ReadoutChannel(0), 1e-6, name='ro', clbit=0), 100e-9),
    ], bias_overrides={0: .1})
    sweep = Sweep.grid(SweepAxis(amp, [.1, .2]), SweepAxis(duration, [20e-9, 40e-9]))
    expanded = sweep.expand(program)
    rows = [program.metadata["sweep"] for program in expanded]
    for bound, row in zip(expanded, rows):
        assert bound[0].pulse.envelope.amplitude == 2 * row['amp']
        assert bound[0].pulse.duration == row['duration']
        assert [entry.start for entry in bound] == [10e-9, 100e-9]
        assert bound[1].operation.name == 'ro'
        assert bound[1].operation.clbit == 0
        assert bound.bias_overrides == {0: .1}
        assert not bound.parameters
    assert program.parameters == {amp, duration}


@pytest.mark.parametrize('mode, supplied, expected', [('absolute', .5, [.5, .5]),
    ('relative', 2, [.4, .8]), ('delta', .1, [.3, .5])])
def test_schedule_modes_and_per_qubit(mode, supplied, expected):
    amp = Parameter('amplitude', group='calibration', gate='x')
    program = PulseSchedule([ScheduledPulse(DriveChannel(q), Pulse(Constant(20e-9, amp)), 0)
                             for q in (2, 4)])
    params = CalibrationParams()
    for q, baseline in [(2, .2), (4, .4)]:
        params = params.set(qubit=(q,), parameter='amplitude', gate='x', value=baseline)
    target = Target(inst_sched_reg=InstSchedRegistry([2, 4]), calibration_params=params)
    sweep = Sweep.zip(SweepAxis(amp, {2: [supplied], 4: [supplied]}, value_mode=mode))
    expanded = sweep.expand(program, target=target)
    np.testing.assert_allclose([e.pulse.envelope.amplitude for e in expanded[0]], expected)


def test_schedule_unbound_parameter_rejected():
    a, b = Parameter('a'), Parameter('b')
    program = PulseSchedule([ScheduledPulse(DriveChannel(0), Pulse(Constant(b, a)), 0)])
    with pytest.raises(ValueError, match='Unbound'):
        Sweep.grid(SweepAxis(a, [.2])).expand(program)


def test_symbolic_starts_and_expressions_bind_for_pulses_and_acquisitions():
    start = Parameter('start')
    program = PulseSchedule([
        ScheduledPulse(DriveChannel(0), Pulse(Constant(20e-9, .2)), start),
        ScheduledAcquire(Acquire(ReadoutChannel(0), 100e-9, name='ro'), start + 30e-9),
    ])
    assert program.parameters == {start}
    expanded = Sweep.grid(SweepAxis(start, [10e-9, 50e-9])).expand(program)
    for bound, value in zip(expanded, [10e-9, 50e-9]):
        assert bound[0].start == value
        assert bound[1].start == value + 30e-9
        assert bound.duration == pytest.approx(value + 130e-9)
        assert not bound.parameters
        np.testing.assert_allclose(bound.sample(DriveChannel(0), [value + 10e-9]), [.2])
    assert program[0].start is start


def test_per_qubit_start_binding():
    start = Parameter('start')
    program = PulseSchedule([ScheduledPulse(DriveChannel(q), Pulse(Constant(20e-9, .2)), start)
                             for q in (2, 4)])
    expanded = Sweep.grid(SweepAxis(start, {2: [10e-9], 4: [30e-9]})).expand(program)
    assert [entry.start for entry in expanded[0]] == [10e-9, 30e-9]


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
@pytest.mark.parametrize('acquire', [False, True])
def test_bound_starts_are_validated(value, acquire):
    start = Parameter('start')
    entry = (ScheduledAcquire(Acquire(ReadoutChannel(0), 20e-9), start) if acquire else
             ScheduledPulse(DriveChannel(0), Pulse(Constant(20e-9, .2)), start))
    with pytest.raises(ValueError, match='start'):
        Sweep.grid(SweepAxis(start, [value])).expand(PulseSchedule([entry]))
