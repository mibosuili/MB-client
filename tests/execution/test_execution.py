import numpy as np
import pandas as pd
import pytest

from mb_client.common import Parameter
from mb_client.circuit import QuantumCircuit
from mb_client.compilation import CalibrationParams, InstSchedRegistry, Target
from mb_client.execution import Sweep, SweepAxis
from mb_client.execution.sweep import BatchPreparation

def test_relative_delta_and_per_qubit_sweeps_use_snapshot():
    amplitude = Parameter('amplitude', group='calibration')
    circuit = QuantumCircuit(2)
    circuit.physical_qubits = (2, 4)
    circuit.x(0, amplitude=amplitude)
    circuit.x(1, amplitude=amplitude)
    registrations = InstSchedRegistry([2, 4])
    target = Target(inst_sched_reg=registrations, calibration_params=
        CalibrationParams(
            pd.DataFrame([
                dict(qubit=(2,), parameter='amplitude', value=.2),
                dict(qubit=(4,), parameter='amplitude', value=.4),
            ])))
    preparation = BatchPreparation(target)
    target.set_calibration_params(CalibrationParams())
    sweep = Sweep.grid(SweepAxis(amplitude, [1., 2.], value_mode='relative'))
    circuits = preparation.expand([circuit, circuit], sweep)
    rows = [program.metadata["sweep"] for program in circuits]
    assert [i.operation.calibration['amplitude']
            for i in circuits[2]] == [.4, .8]
    assert circuit.parameters == {amplitude}
    sweep = Sweep.zip(
        SweepAxis(amplitude, {
            2: [.1],
            4: [.2]
        }, value_mode='delta'))
    circuits = preparation.expand([circuit], sweep)
    rows = [program.metadata["sweep"] for program in circuits]
    np.testing.assert_allclose(
        [i.operation.calibration['amplitude'] for i in circuits[0]], [.3, .6])
    with pytest.raises(ValueError, match='does not match'):
        preparation.expand([circuit],
                           Sweep.grid(SweepAxis(amplitude, {2: [1.]})))


def test_conflicting_sweep_calibrations_are_rejected():
    amplitude = Parameter('amplitude', group='calibration')
    circuit = QuantumCircuit(1)
    circuit.x(0, amplitude=amplitude)
    circuit.sx(0, amplitude=amplitude)
    registrations = InstSchedRegistry([0])
    target = Target(inst_sched_reg=registrations, calibration_params=
        CalibrationParams(
            pd.DataFrame([
                dict(qubit=(0,), parameter='amplitude', value=.2, gate='x'),
                dict(qubit=(0,), parameter='amplitude', value=.1, gate='sx'),
            ])))
    with pytest.raises(ValueError, match='conflicting'):
        BatchPreparation(target).expand(
            [circuit],
            Sweep.grid(SweepAxis(amplitude, [1.], value_mode='relative')))