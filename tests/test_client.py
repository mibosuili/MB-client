"""Tests for client experiment submission, result parsing, and export."""

import pytest

from mb_client import (ExecutionResult, Job, JobStatus, QuantumCircuit,
                             RemoteBackend, decode_circuit, encode_circuit)
from mb_client.execution import Sweep, SweepAxis

def test_circuit_protocol_round_trip_and_submission(client, fake_session):
    """Verify that circuits use the safe JSON submission protocol and core operations round-trip without loss."""
    circuit = QuantumCircuit(2, 2, name="bell")
    circuit.physical_qubits = (1, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    payload = encode_circuit(circuit)
    restored = decode_circuit(payload)
    job = client.run(circuit, device="teaching-chip", shots=1000,
                     metadata={"label": "bell"})

    assert isinstance(job, Job)
    assert job.id == "job-1"
    assert job.job_id == "job-1"
    assert job.status() is JobStatus.COMPLETED
    assert restored.name == "bell"
    assert restored.physical_qubits == (1, 2)
    assert [item.operation.name for item in restored] == [
        "h", "cx", "measure", "measure"
    ]
    submitted = fake_session.submissions[-1]
    assert submitted["metadata"] == {"label": "bell"}
    assert submitted["experiment"] == "circuit"
    assert submitted["programs"] == [payload]
    assert submitted["run_options"] == {
        "shots": 1000,
        "measure_level": "state",
    }
    result = job.result()
    assert isinstance(result, ExecutionResult)
    assert result.program_results[0].counts == {"00": 490, "11": 510}


def test_circuit_protocol_rejects_untrusted_operations():
    """Verify that the protocol does not load arbitrary Python types from external input."""
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    payload = encode_circuit(circuit)
    payload["operations"][0]["name"] = "__import__"

    with pytest.raises(ValueError, match="Unsupported circuit operation"):
        decode_circuit(payload)

    payload["operations"] = [{
        "name": "id", "qubits": [], "clbits": [], "params": []
    }]
    with pytest.raises(ValueError, match="id requires"):
        decode_circuit(payload)


def test_device_bound_backend_reuses_device(client, fake_session):
    """Verify that a bound device need not be supplied for every circuit submission."""
    circuit = QuantumCircuit(1, 1)
    circuit.physical_qubits = (1,)
    circuit.x(0)
    circuit.measure(0, 0)

    backend = client.backend("teaching-chip")
    backend.run([circuit], shots=20)

    assert isinstance(backend, RemoteBackend)
    assert fake_session.submissions[-1]["device"] == "teaching-chip"
    assert fake_session.submissions[-1]["run_options"]["shots"] == 20
