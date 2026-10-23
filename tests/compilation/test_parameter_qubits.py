import pytest

from mb_client.common import Parameter
from mb_client.circuit import CXGate
from mb_client.compilation import CalibrationParams, InstSchedRegistry, Target
from mb_client.pulse import Constant, DriveChannel, Pulse, PulseSequence


@pytest.mark.parametrize('qubits', [(), (True,), (-1,), (1, 1), [1], 'q1'])
def test_invalid_parameter_qubits(qubits):
    with pytest.raises(ValueError, match='qubits'):
        Parameter('amplitude', group='calibration', qubits=qubits)


def test_selectors_are_calibration_only_and_old_fields_removed():
    with pytest.raises(ValueError, match='operation'):
        Parameter('amplitude', qubits=(17,))
    for field, value in [('operand', 0), ('device', 'q17')]:
        with pytest.raises(TypeError):
            Parameter('amplitude', group='calibration', **{field: value})


@pytest.mark.parametrize('selector, expected', [(None, .4), ((17,), .2), ((18, 17), .6)])
def test_pair_inference_and_explicit_qubits(selector, expected):
    amplitude = Parameter('amplitude', group='calibration', qubits=selector)
    template = PulseSequence(template=True)
    template.play(DriveChannel(0), Pulse(Constant(20e-9, amplitude)))
    registry = InstSchedRegistry([17, 18])
    registry.add(CXGate(), (17, 18), template)
    params = CalibrationParams()
    for qubits, value in [((17, 18), .4), ((17,), .2), ((18, 17), .6)]:
        params = params.set(qubit=qubits, parameter='amplitude', gate='cx', value=value)
    target = Target(inst_sched_reg=registry, calibration_params=params)
    resolved = target.resolve(CXGate(), (17, 18)).schedule()
    assert resolved[0].pulse.envelope.amplitude == expected
    assert target.get_calibration_param((17, 18), 'amplitude', gate='cx') == .4
    assert target.get_calibration_param((17,), 'amplitude', gate='cx') == .2


@pytest.mark.parametrize('key', ['q17', 17, [17], (), (17, 17), (True,)])
def test_calibration_table_rejects_noncanonical_keys(key):
    import pandas as pd
    with pytest.raises(ValueError, match='tuple'):
        CalibrationParams().set(qubit=key, parameter='amplitude', value=.2)
    with pytest.raises(ValueError, match='tuple'):
        CalibrationParams(pd.DataFrame([dict(qubit=key, parameter='amplitude', value=.2)]))
    with pytest.raises(ValueError, match='tuple'):
        CalibrationParams().get(key, 'amplitude')


def test_tuple_table_round_trip_and_registry_defaults():
    params = CalibrationParams().set(qubit=(17,), parameter='amplitude', value=.2)
    params = params.set(qubit=(17, 18), parameter='amplitude', value=.4)
    restored = CalibrationParams.from_dataframe(params.to_dataframe())
    assert restored.get((17,), 'amplitude') == .2
    assert restored.get((17, 18), 'amplitude') == .4
    assert list(restored.to_dataframe().qubit) == [(17,), (17, 18)]
    template = PulseSequence(template=True)
    template.play(DriveChannel(0), Pulse(Constant(20e-9, Parameter(
        'amplitude', group='calibration', default=.4))))
    registry = InstSchedRegistry([17, 18])
    registry.add(CXGate(), (17, 18), template)
    assert registry.default_calibration_params().get((17, 18), 'amplitude', gate='cx') == .4
