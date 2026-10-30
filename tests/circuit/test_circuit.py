import pytest

from mb_client.circuit import Gate, QuantumCircuit


def test_append_copies_operation_and_validates_arity():
    circuit = QuantumCircuit(2)
    gate = Gate("pair", 2, [0.5])
    item = circuit.append(gate, [0, 1])
    gate.params[0] = 1.0

    assert item.operation.params == [0.5]
    with pytest.raises(ValueError, match="expects"):
        circuit.append(gate, [0])


def test_circuit_to_gate_and_compose():
    body = QuantumCircuit(1, name="body")
    body.x(0)
    logical_gate = body.to_gate()
    circuit = QuantumCircuit(2)
    circuit.append(logical_gate, [1])

    expanded = QuantumCircuit(2).compose(body, qubits=[1])

    assert logical_gate.definition is not body
    assert circuit.data[0].qubits[0].index == 1
    assert expanded.data[0].operation.name == "x"
    assert expanded.data[0].qubits[0].index == 1


def test_measure_uses_classical_bit_and_barrier_defaults_to_all_qubits():
    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0)
    circuit.barrier()

    assert circuit.data[0].operation.name == "measure"
    assert circuit.data[0].clbits[0].index == 0
    assert circuit.data[1].operation.name == "barrier"
    assert len(circuit.data[1].qubits) == 2


def test_reset_accepts_multiple_qubits_and_brackets_them_with_alignment():
    circuit = QuantumCircuit(3)

    resets = circuit.reset(0, 2)

    assert isinstance(resets, tuple)
    assert [item.operation.name for item in circuit.data] == [
        "barrier", "reset", "reset", "barrier"
    ]
    assert [qubit.index for qubit in circuit.data[0].qubits] == [0, 2]
    assert [qubit.index for qubit in circuit.data[-1].qubits] == [0, 2]
