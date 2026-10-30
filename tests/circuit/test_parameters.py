import pytest

from mb_client.common import Parameter
from mb_client.circuit import QuantumCircuit


def test_parameter_groups_are_validated():
    assert Parameter("amp", group="calibration").group == "calibration"
    with pytest.raises(ValueError, match="group"):
        Parameter("amp", group="unknown")


def test_circuit_discovers_and_immutably_binds_parameters():
    delay = Parameter("delay")
    amplitude = Parameter("amplitude", group="calibration")
    circuit = QuantumCircuit(1)
    circuit.delay(delay / 2, 0)
    circuit.x(0)
    circuit.data[-1].operation.calibration["amplitude"] = amplitude

    bound = circuit.bind_parameters({delay: 2e-6, amplitude: 0.25})

    assert circuit.parameters == {delay, amplitude}
    assert bound.parameters == set()
    assert bound.data[0].operation.params[0] == 1e-6
    assert bound.data[1].operation.calibration["amplitude"] == 0.25
    assert circuit.data[1].operation.calibration["amplitude"] == amplitude


def test_partial_binding_leaves_remaining_parameters():
    first = Parameter("first")
    second = Parameter("second")
    circuit = QuantumCircuit(1)
    circuit.delay(first + second, 0)

    bound = circuit.bind_parameters({first: 1.0})

    assert bound.parameters == {second}


def test_unknown_parameter_binding_is_rejected():
    circuit = QuantumCircuit(1)
    with pytest.raises(ValueError, match="Unknown circuit parameters"):
        circuit.bind_parameters({"missing": 1})
