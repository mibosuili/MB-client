---
description: The Python SDK for circuits, pulse exploration, and remote quantum experiments.
---

# Quantum experiments, from Python

**MB Client** is the Python SDK from Mibo SLabs. Construct circuits locally,
inspect pulse models, and submit experiments to an available MB Cloud backend.

One vocabulary connects the workflow: **QuantumCircuit → Backend → Job → ExecutionResult**.

[GitHub source](https://github.com/mibosuili/MB-client) ·
[Install from PyPI](https://pypi.org/project/mibo-client/)

<div class="grid cards" markdown>

- **Start with a circuit**

    Build and draw your first circuit without a server or API key.

    [Open the quickstart](getting-started/quickstart.md)

- **Run a remote experiment**

    Submit a small, deterministic test to a production simulator resource.

    [Try the simulator example](examples/production-simulator.md)

- **Understand the results**

    Learn about shots, state counts, IQ arrays, and batch results.

    [Read the results guide](guides/jobs-and-results.md)

- **Explore the API**

    Browse signatures and docstrings generated from the current SDK source.

    [Browse MBCloudClient](api/client.md)

</div>

## A circuit does not require a connection

```python
from mb_client import QuantumCircuit

circuit = QuantumCircuit(1, 1, name="prepare-one")
circuit.x(0)
circuit.measure(0, 0)
print(circuit.draw())
```

Construction and drawing are local. Executing a job requires an online resource
and an API key for the service you connect to.

## What runs where?

| Component | Responsibility |
| --- | --- |
| Client | Circuit construction, local compilation and visualization, HTTPS requests |
| Server | Authentication, validation, resource selection, job scheduling, stored results |
| Simulator worker | State-vector simulation and shot sampling on CPU or GPU |
| Hardware worker | Device-specific compilation and instrument execution |

Client users do not need to connect to Redis or MongoDB, install instrument
drivers, or deploy a worker. See [Installation](getting-started/installation.md)
for supported Python versions.

!!! note "Local capabilities and remote capabilities are different"
    The SDK can represent custom gates and pulse schedules locally. The cloud
    circuit protocol accepts only a supported operation subset; an individual
    backend may impose additional restrictions. A successful local compilation
    does not imply that a remote resource can execute the program.
