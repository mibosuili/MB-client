# Quickstart

## 1. Construct and inspect a circuit locally

This code does not make a network request:

```python
from mb_client import QuantumCircuit, encode_circuit

circuit = QuantumCircuit(1, 1, name="prepare-one")
circuit.physical_qubits = (0,)
circuit.x(0)
circuit.measure(0, 0)

print(circuit.draw())
print(encode_circuit(circuit))
```

`QuantumCircuit(1, 1)` allocates one logical qubit and one classical bit.
The X gate changes the initial state from `|0>` to `|1>`; measurement writes the
outcome to classical bit 0. `physical_qubits = (0,)` selects resource qubit `q0`.

## 2. Connect to MB Cloud

Create an API key in the production web console, then replace the placeholder:

```python
from mb_client import MBCloudClient

client = MBCloudClient(
    base_url="https://cloud.mbslabs.tech",
    api_key="YOUR_PRODUCTION_API_KEY",
)

print(client.ping())
for resource in client.resources():
    print(resource["id"], resource["status"])
```

Use an online resource returned for your account. A local-development API key
is not automatically valid in production.

## 3. Submit and read the result

The [complete production example](../examples/production-simulator.md) combines
connection, circuit construction, submission, and error handling in one script.
With `client` and `circuit` from the steps above, replace the resource placeholder
with the ID of an online simulator returned by `client.resources()`:

```python
job = client.run(
    circuit,
    device="YOUR_SIMULATOR_RESOURCE",
    shots=1000,
    measure_level="state",
)
print("Job:", job.id)

result = job.wait(timeout=120, polling_interval=2.0)
print(result.program_results[0].counts)
```

For this ideal-simulator example, the expected counts are `{'1': 1000}`.
Actual hardware results can differ because of noise, calibration, and readout.

!!! warning "Submitting executes a remote job"
    Unlike constructing a circuit, `client.run()` submits work to the selected
    service. Verify the resource identifier before using a hardware backend.
