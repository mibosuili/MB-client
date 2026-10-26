import json

import numpy as np
import pytest

from mb_client import (
    ExecutionResult, MeasurementLevel, ProgramResult, QuantumCircuit, RunOptions,
    decode_circuit, decode_result, encode_circuit, encode_result,
)
from mb_client.protocol import SUPPORTED_OPERATIONS


def wire(value):
    return json.loads(json.dumps(value, allow_nan=False))


@pytest.mark.parametrize('level', list(MeasurementLevel))
@pytest.mark.parametrize('name', [None, '', 'experiment'])
def test_result_round_trip(level, name):
    data = {'0': 2} if level is MeasurementLevel.STATE else np.array([[1+2j], [3-4j]])
    result = ExecutionResult(
        job_id='job-1', backend_name='chip', experiment_name=name,
        metadata={'label': ['test']},
        run_options=RunOptions(shots=2, measure_level=level,
                               iq_centers={3: [0j, 1+2j]}),
        program_results=tuple(ProgramResult(
            index=i, input_index=0, program_type=kind, name='program',
            physical_qubits=(3,), shots=2, measure_level=level, data=data,
            metadata={'sweep': {'angle': i}},
            runtime_info={'scalar': 2+3j, 'empty': np.empty((0, 2))},
        ) for i, kind in enumerate(('circuit', 'pulse_schedule'))),
        runtime_info={'nested': [{'array': np.array(2+1j)}]},
    )
    payload = wire(encode_result(result))
    restored = decode_result(payload)
    assert encode_result(restored) == payload
    assert restored.experiment_name == name
    assert restored.run_options == result.run_options
    assert restored.program_results[1].coordinates == {'angle': 1}
    for record in restored.program_results:
        if level is MeasurementLevel.STATE:
            assert record.counts == data
        else:
            np.testing.assert_array_equal(record.data, data)
        assert record.runtime_info['scalar'] == 2+3j
        assert record.runtime_info['empty'].shape == (0, 2)


def test_circuit_round_trip_all_supported_operations():
    circuit = QuantumCircuit(2, 2, name='all operations')
    circuit.physical_qubits = (3, 7)
    circuit.metadata = {'nested': {'labels': ['test']}}
    for name in ('x', 'y', 'sx', 'sy', 'h'):
        getattr(circuit, name)(1)
    for name in ('rx', 'ry', 'rz'):
        getattr(circuit, name)(0.25, 1)
    circuit.rxy(0.2, 0.3, 0)
    circuit.id(2e-8, 1)
    circuit.id(None, 0)
    circuit.delay(1e-8, 1, 'drive')
    circuit.cx(1, 0)
    circuit.cz(0, 1)
    circuit.barrier(0, 1)
    circuit.reset(1, align=False)
    circuit.measure(1, 0)
    payload = wire(encode_circuit(circuit))
    assert {op['name'] for op in payload['operations']} == SUPPORTED_OPERATIONS
    restored = decode_circuit(payload)
    assert encode_circuit(restored) == payload
    assert restored.physical_qubits == (3, 7)
    assert restored.metadata == circuit.metadata


def test_empty_circuit_round_trip():
    payload = wire(encode_circuit(QuantumCircuit(1)))
    assert encode_circuit(decode_circuit(payload)) == payload
