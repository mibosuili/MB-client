# Circuits and qubits

## Logical, resource, and classical bits

```python
from mb_client import QuantumCircuit

circuit = QuantumCircuit(2, 2, name="bell")
circuit.physical_qubits = (0, 1)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)
print(circuit.draw())
```

| Concept | Meaning in this example |
| --- | --- |
| Logical qubit | Circuit operand 0 or 1 passed to a gate |
| Resource qubit | `physical_qubits[logical_index]`; `(0, 1)` selects `q0`, `q1` |
| Classical bit | Destination for a measurement outcome |
| Backend identifier | Resource slug passed as `device`, not a qubit identifier |

For real hardware, choose available physical indices from `client.resources()`.
Do not assume every chip exposes consecutive qubits starting at zero.

## Gate and measurement operations

Frequently used operations include `x`, `y`, `h`, `rx`, `ry`, `rz`, `cx`, `cz`,
`measure`, and `measure_all`. Gate angles are in radians. Timing values such as
delay durations use seconds.

The cloud protocol currently encodes `barrier`, `cx`, `cz`, `delay`, `h`, `id`,
`measure`, `reset`, `rx`, `ry`, `rxy`, `rz`, `sx`, `sy`, `x`, and `y`. This is a
serialization whitelist, not a promise that every backend executes every entry.
For example, the current simulator accepts terminal state measurements but
does not support IQ, reset, or arbitrary pulse schedules.

!!! warning "Check both server and worker capabilities"
    Local circuit methods, the server validator, and a backend compiler can have
    different support sets. In particular, delay-channel vocabulary changed in
    recent SDK versions; confirm server and hardware-adapter compatibility before
    using delay operations on a hardware resource.

## Bind parameters before remote execution

```python
from mb_client import QuantumCircuit, encode_circuit
from mb_client.common import Parameter

theta = Parameter("theta")
template = QuantumCircuit(1, 1, name="rotation")
template.rx(theta, 0)
template.measure(0, 0)

bound = template.bind_parameters({theta: 0.25})
payload = encode_circuit(bound)
assert payload["operations"][0]["params"] == [0.25]
```

Symbolic or custom local operations are not executable wire objects. Bind all
parameters and translate unsupported custom gates before submitting them.

## Drawing

`circuit.draw()` returns text. `circuit.draw(format="mpl")` returns a Matplotlib
figure; pass `show=True` to display it immediately in an interactive session.
See the notebook examples in the checkout under `examples/visualization/` for
styles, export, and richer drawings.

See [QuantumCircuit API](../api/circuits.md) for current method signatures.
