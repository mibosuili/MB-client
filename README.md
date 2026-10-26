# MB Cloud Client

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python versions: 3.10, 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)

A unified quantum SDK for experimental users, providing circuit construction,
pulse compilation, result analysis, and remote job interfaces for MB Cloud.

The SDK uses unified `Backend`, `Job`, and `ExecutionResult` interfaces to represent
backends, jobs, and results.

## Current Availability

**Simulator and hardware backends are still being deployed. Both are currently unavailable, so job submission and execution are not yet supported.**

Circuit construction, custom gates, basis translation, pulse sequence construction,
circuit-to-pulse compilation, and visualization are currently available locally.
The examples below demonstrate these local features and require no backend connection
or API key. Compiling a pulse schedule does not mean it has been executed on a simulator
or hardware.

## Installation

Supports Python 3.10 and 3.11 (`>=3.10,<3.12`).

Run from the repository root:

```bash
python -m pip install .
```

## Development

Run from the repository root:

```bash
python -m pip install -e .
```

## Examples

Examples are provided as Jupyter notebooks in [`examples/`](examples/).
Start with the basic examples in the order below, then explore the visualization examples.

| Example | Description |
| --- | --- |
| [Circuit construction](examples/basic/circuit_construction.ipynb) | Built-in gates, custom gates defined from subcircuits or `Gate` subclasses, and unitary gates represented as matrices. |
| [Basis translation](examples/basic/transpile_unitary_gates.ipynb) | Use `translate_basis()` to convert gates and circuits to the `rz`, `sx`, `x`, and `cz` basis gates, without routing or pulse scheduling. |
| [Pulse construction](examples/basic/pulse_construction.ipynb) | Build a reset → X → measurement pulse sequence for q1 using both direct construction and `with PulseSequence()`, then compare and plot the schedules. |
| [Circuit-to-pulse compilation](examples/basic/circuit_to_pulse_compilation.ipynb) | Define pulse templates, an instruction registry, and calibration parameters from scratch, build a `Target`, and use `schedule()` to compile a reset → X → delay → measurement circuit directly into a `PulseSchedule`. |
| [Circuit drawing](examples/visualization/circuit_drawing.ipynb) | Text and Matplotlib drawings, with dictionary and JSON style configuration. |
| [Pulse drawing](examples/visualization/pulse_drawing.ipynb) | Visualize pulse envelopes, independent channel schedules, delays, barriers, and waveforms. |

## Backend Connection Configuration (Once the Service Is Available)

Once backend services are available, configure the connection in a `.env` file
in your working directory:

```dotenv
MB_CLOUD_API_KEY=mbc_your_api_key
MB_CLOUD_BASE_URL=https://cloud.mbslabs.tech
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
