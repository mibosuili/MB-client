# Production Simulator example

Submit a single-qubit circuit to an online production simulator resource. This
is a simulator example, not a physical-hardware experiment.

## Prerequisites

- [Install the client](../getting-started/installation.md) on Python 3.10–3.12.
- Create a production API key at [MB Cloud](https://cloud.mbslabs.tech/).
- Query `client.resources()` and select an online simulator with `q0` available.
- Copy its exact resource ID into `RESOURCE` below. Resource names, qubit limits,
  and account access can change; do not assume a particular resource is available.

## Complete script

Save as `test_simulator_prod.py`, replace the API key and resource placeholders, and run it from
your client environment. Explicit values take precedence over `.env` settings.

```python
from mb_client import MBCloudClient, QuantumCircuit
from mb_client.exceptions import ServerConnectionError
from mb_client.execution import JobCancelledError, JobExecutionError


BASE_URL = "https://cloud.mbslabs.tech"
API_KEY = "YOUR_PRODUCTION_API_KEY"
RESOURCE = "YOUR_SIMULATOR_RESOURCE"


def main():
    if API_KEY == "YOUR_PRODUCTION_API_KEY" or RESOURCE == "YOUR_SIMULATOR_RESOURCE":
        raise SystemExit("Replace the API key and resource placeholders before submitting a job.")

    client = MBCloudClient(base_url=BASE_URL, api_key=API_KEY)
    print("Server:", client.base_url)
    print("Health:", client.ping())

    circuit = QuantumCircuit(1, 1, name="simulator-production-test")
    circuit.physical_qubits = (0,)
    circuit.x(0)
    circuit.measure(0, 0)

    job = client.run(circuit, device=RESOURCE, shots=1000, measure_level="state")
    print("Job:", job.id)
    try:
        result = job.wait(timeout=120, polling_interval=2.0)
    except TimeoutError:
        print("Still waiting. Keep this job ID and query it before resubmitting:", job.id)
        return
    except (JobExecutionError, JobCancelledError):
        print("Job details:", job.details())
        raise

    program = result.program_results[0]
    print("Backend:", result.backend_name)
    print("Shots:", program.shots)
    print("Counts:", program.counts)
    assert program.counts == {"1": 1000}


if __name__ == "__main__":
    try:
        main()
    except ServerConnectionError as exc:
        raise SystemExit(f"MB Cloud API request failed: {exc}") from exc
```

```bash
python test_simulator_prod.py
```

## Expected output

```text
Server: https://cloud.mbslabs.tech
Health: pong
Job: <job UUID>
Backend: <your simulator resource ID>
Shots: 1000
Counts: {'1': 1000}
```

The X gate prepares `|1>` from `|0>`, so the ideal simulator returns 1 in all
1,000 shots. Find the same job ID in the production console to inspect its state
and result. This assertion is specific to an ideal simulator, not noisy hardware.

!!! warning "Credentials in public examples"
    Documentation deliberately uses a placeholder, not a live production key.
    Do not publish a real key when sharing your script. For applications, prefer
    [environment configuration](../getting-started/configuration.md).

## If the job stays queued

Keep the job ID and ask the operator to check simulator availability and queue
routing. Client users do not configure worker queues or Redis connections.
