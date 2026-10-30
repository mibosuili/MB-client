# Pulses and compilation

Pulse construction and compilation are local SDK features. They are useful for
exploring measurement/control concepts without connecting to physical hardware.

## Channels

| Channel | Concept |
| --- | --- |
| `DriveChannel(q)` | Drive the selected qubit |
| `FluxChannel(q)` | Flux control for the selected qubit |
| `ReadoutChannel(q)` | Readout and acquisition-related operations |

Channel indices identify resource qubits. A channel object does not open a
connection to an instrument.

## Construct a pulse sequence locally

```python
from mb_client.pulse import Constant, DriveChannel, Pulse, PulseSequence

drive = DriveChannel(0)
sequence = PulseSequence()
sequence.play(drive, Pulse(Constant(duration=20e-9, amplitude=0.1)))
sequence.delay(drive, 10e-9)

schedule = sequence.schedule()
print(schedule)
```

This example schedules abstract pulse operations. It does not calibrate an
instrument, generate a hardware trigger, or execute a measurement.

## Circuit-to-pulse compilation

Compilation combines an instruction registry and calibration parameters in a
`Target`. The `schedule()` compiler lowers a circuit against that target and
constructs a `PulseSchedule`. `translate_basis()` instead translates gates into
a chosen basis; it does not perform routing or hardware execution.

Use the checked-in notebooks for complete, self-contained definitions:

- `examples/basic/pulse_construction.ipynb`
- `examples/basic/circuit_to_pulse_compilation.ipynb`
- `examples/basic/transpile_unitary_gates.ipynb`
- `examples/visualization/pulse_drawing.ipynb`

!!! note "The remote API currently accepts circuits only"
    `MBCloudClient.submit()` rejects `PulseSchedule` objects. A pulse API in the
    local SDK does not imply a corresponding cloud submission endpoint. Hardware
    execution and instrument calibration are handled by the hardware worker.

See [Pulse and compilation API](../api/compilation.md) for signatures.
