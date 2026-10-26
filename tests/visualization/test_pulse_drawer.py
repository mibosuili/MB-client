import matplotlib.pyplot as plt
import numpy as np
import pytest

from mb_client.pulse import Constant, DriveChannel, FluxChannel, Pulse, PulseSequence, ReadoutChannel


def test_waveform_plot_and_text_delegates():
    sequence = PulseSequence()
    sequence.play(DriveChannel(0), Pulse(Constant(20e-9, .3)))
    sequence.acquire(ReadoutChannel(0), 1e-6)
    schedule = sequence.schedule()
    assert 'amplitude=0.3' in str(sequence)
    assert 'Acquire' in str(schedule)
    figure, axes = schedule.draw(sample_rate=1e8)
    assert axes.shape == (1,)
    assert axes[0].get_xlabel() == 'Time (ns)'
    plt.close(figure)


def test_flux_q_component_is_rejected_by_renderer():
    sequence = PulseSequence()
    sequence.play(FluxChannel(0), Pulse(Constant(20e-9, .3), phase=1))
    with pytest.raises(ValueError, match='nonzero Q'):
        sequence.schedule().draw(sample_rate=1e8)
    plt.close('all')


@pytest.mark.parametrize('channel_type', [DriveChannel, FluxChannel, ReadoutChannel])
def test_long_pulses_are_compressed(channel_type):
    sequence = PulseSequence()
    channel = channel_type(0)
    sequence.play(channel, Pulse(Constant(1e-6, .3)))
    sequence.delay(channel, 20e-9)
    sequence.play(channel, Pulse(Constant(2e-6, .2)))
    schedule = sequence.schedule()
    figure, axes = schedule.draw(max_pulse_duration=100e-9)
    np.testing.assert_allclose(axes[0].get_xlim(), [0, 220])
    np.testing.assert_allclose(axes[0].get_xticks(), [0, 100, 120, 220])
    assert not axes[0].patches
    plt.close(figure)
    figure, axes = schedule.draw(max_pulse_duration=None)
    np.testing.assert_allclose(axes[0].get_xlim(), [0, 3020])
    assert not axes[0].patches
    plt.close(figure)


def test_simultaneous_long_pulses_share_compression():
    sequence = PulseSequence()
    for index in (0, 1):
        sequence.play(DriveChannel(index), Pulse(Constant(1e-6, .3)))
    figure, axes = sequence.schedule().draw(max_pulse_duration=100e-9)
    for axis in axes:
        np.testing.assert_allclose(axis.get_xlim(), [0, 100])
        assert not axis.patches
    plt.close(figure)


def test_long_pulse_scaling_is_continuous_without_markers():
    sequence = PulseSequence()
    sequence.play(FluxChannel(0), Pulse(Constant(1e-6, .3)))
    times = np.linspace(0, 1e-6, 101)
    figure, axes = sequence.schedule().draw(t=times, max_pulse_duration=100e-9)
    axis = axes[0]
    np.testing.assert_allclose(axis.lines[0].get_xdata(), np.linspace(0, 100, 101))
    assert len(axis.lines) == 2  # Waveform and zero baseline only.
    assert not axis.patches
    plt.close(figure)


@pytest.mark.parametrize('duration', [0, -1, float('inf'), float('nan')])
def test_invalid_max_pulse_duration(duration):
    sequence = PulseSequence()
    sequence.play(DriveChannel(0), Pulse(Constant(20e-9, .3)))
    with pytest.raises(ValueError, match='max_pulse_duration'):
        sequence.schedule().draw(max_pulse_duration=duration)


@pytest.mark.parametrize('first_end, second_start, second_end', [
    (20000, 20000, 25000),
    (20000, 18000, 27000),
])
@pytest.mark.parametrize('target_ns', [100, 200])
def test_connected_pulses_share_one_scale(first_end, second_start, second_end,
                                         target_ns):
    sequence = PulseSequence()
    sequence.play(FluxChannel(0), Pulse(Constant(first_end * 1e-9, .3)))
    sequence.delay(FluxChannel(1), second_start * 1e-9)
    sequence.play(FluxChannel(1),
                  Pulse(Constant((second_end - second_start) * 1e-9, .2)))
    times = np.linspace(0, second_end * 1e-9, 101)
    figure, axes = sequence.schedule().draw(
        t=times, max_pulse_duration=target_ns * 1e-9)
    expected = np.linspace(0, min(second_end, target_ns), 101)
    for axis in axes:
        np.testing.assert_allclose(axis.lines[0].get_xdata(), expected)
        assert len(axis.lines) == 2
    plt.close(figure)


def test_short_gates_do_not_extend_or_bridge_scaled_ranges():
    sequence = PulseSequence()
    channel = FluxChannel(0)
    for duration in (20, 1000, 20, 1000, 20):
        sequence.play(channel, Pulse(Constant(duration * 1e-9, .3)))
    times = np.array([0, 20, 1020, 1040, 2040, 2060]) * 1e-9
    figure, axes = sequence.schedule().draw(t=times, max_pulse_duration=100e-9)
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(),
                               [0, 20, 120, 140, 240, 260])
    plt.close(figure)


def test_shared_endpoint_warning_is_channel_specific():
    from mb_client.pulse.schedule import PulseSchedule, ScheduledPulse
    import warnings
    pulse = Pulse(Constant(20e-9, .3))
    schedule = PulseSchedule([
        ScheduledPulse(DriveChannel(0), pulse, 0),
        ScheduledPulse(DriveChannel(0), Pulse(Constant(20e-9, .2)), 20e-9),
        ScheduledPulse(DriveChannel(1), pulse, 20e-9),
    ])
    with pytest.warns(UserWarning, match='share an endpoint at 20 ns'):
        figure, _ = schedule.draw()
    plt.close(figure)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        figure, _ = schedule.draw(channels=[DriveChannel(1)])
    assert not caught
    plt.close(figure)


def test_identical_shared_endpoints_do_not_warn():
    import warnings
    sequence = PulseSequence()
    for phase in (0, 0):
        sequence.play(DriveChannel(0), Pulse(Constant(20e-9, .3), phase=phase))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        figure, _ = sequence.schedule().draw()
    assert not caught
    plt.close(figure)


@pytest.mark.parametrize('start', [0, 10e-9, 20e-9])
@pytest.mark.parametrize('field', ['envelope amplitude', 'carrier frequency', 'phase'])
def test_different_overlapping_pulses_warn(start, field):
    from mb_client.pulse.schedule import PulseSchedule, ScheduledPulse
    left = Pulse(Constant(20e-9, .3))
    right = Pulse(Constant(20e-9, .2 if field == 'envelope amplitude' else .3),
                  frequency=1e6 if field == 'carrier frequency' else 0,
                  phase=.5 if field == 'phase' else 0)
    schedule = PulseSchedule([ScheduledPulse(DriveChannel(0), left, 0),
                              ScheduledPulse(DriveChannel(0), right, start)])
    with pytest.warns(UserWarning, match=field):
        figure, _ = schedule.draw(t=[0, 40e-9], with_carrier=False)
    plt.close(figure)


@pytest.mark.parametrize('amplitudes', [(0, .3), (.3, 0), (0, 0)])
@pytest.mark.parametrize('start', [0, 10e-9, 20e-9])
def test_zero_amplitude_overlap_does_not_warn(amplitudes, start):
    import warnings
    from mb_client.pulse.schedule import PulseSchedule, ScheduledPulse
    schedule = PulseSchedule([
        ScheduledPulse(DriveChannel(0), Pulse(Constant(20e-9, amplitudes[0])), 0),
        ScheduledPulse(DriveChannel(0), Pulse(Constant(20e-9, amplitudes[1]),
                                             frequency=1e6, phase=.5), start),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        figure, _ = schedule.draw()
    assert not caught
    plt.close(figure)


@pytest.mark.parametrize('zero_on_left', [True, False])
def test_zero_envelope_at_shared_endpoint_does_not_warn(zero_on_left):
    import warnings
    from mb_client.pulse.envelopes import Cos
    from mb_client.pulse.schedule import PulseSchedule, ScheduledPulse
    envelopes = [Cos(20e-9, .3), Constant(20e-9, .2)]
    if not zero_on_left:
        envelopes.reverse()
    schedule = PulseSchedule([
        ScheduledPulse(DriveChannel(0), Pulse(envelopes[0]), 0),
        ScheduledPulse(DriveChannel(0), Pulse(envelopes[1], frequency=1e6, phase=.5), 20e-9),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        figure, _ = schedule.draw()
    assert not caught
    plt.close(figure)


def test_zero_endpoints_do_not_hide_nonzero_interior_overlap():
    from mb_client.pulse.envelopes import Cos
    from mb_client.pulse.schedule import PulseSchedule, ScheduledPulse
    schedule = PulseSchedule([
        ScheduledPulse(DriveChannel(0), Pulse(Cos(20e-9, .3)), 0),
        ScheduledPulse(DriveChannel(0), Pulse(Cos(20e-9, .2)), 0),
    ])
    with pytest.warns(UserWarning, match='envelope amplitude'):
        figure, _ = schedule.draw(t=[0, 20e-9])
    plt.close(figure)
