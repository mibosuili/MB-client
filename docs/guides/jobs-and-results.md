# Jobs and measurement results

## The job lifecycle

`client.run()` returns a `RemoteJob` after the service accepts a submission.
Acceptance is not the same as completion.

| Status | Meaning |
| --- | --- |
| `queued` | Waiting to be dispatched |
| `running` | Dispatched for execution; inspect details and worker logs if delayed |
| `completed` | Result is available |
| `failed` | Execution failed; the job's error explains why |
| `cancelled` | Cancelled while waiting |

```python
# Given a job returned by client.run(...):
print(job.id)
print(job.status().value)
print(job.details())

result = job.wait(timeout=120, polling_interval=2.0)
```

`job.result()` is an alias of `wait()` on a remote job. A wait timeout does not
cancel the service-side execution. Keep the job ID so you can retrieve it later:

```python
# Given a configured client and a previously saved job_id:
job = client.backend("YOUR_SIMULATOR_RESOURCE").retrieve_job(job_id)
result = job.wait(timeout=120, polling_interval=2.0)
```

Use the same resource ID that was used when submitting the saved job.

Only queued jobs can be cancelled through the current cloud API. A status may
change between checking it and calling `job.cancel()`; inspect the return value.

## STATE: sampled counts

For `measure_level="state"`, each `ProgramResult.counts` is a bit-string-to-count
dictionary. For example, `{'0': 507, '1': 493}` means 507 shots measured 0 and
493 shots measured 1, for a total of 1,000 shots.

The empirical probability of a result is its count divided by the total number
of recorded shots. More shots improve the sampling estimate; they do not remove
systematic hardware errors.

The cloud counts convention prints classical bits in descending order:
`c[n-1] ... c[0]`. If only classical bit 0 of a two-bit result is 1, its key is
`"01"`. Logical-to-classical measurement mapping matters.

## IQ: complex-valued data

IQ is available only on a supporting hardware backend, not the current simulator. The client
decodes JSON real/imaginary arrays into complex-valued data:

```python
# Given a completed IQ result:
program = result.program_results[0]
print(program.data)
print(program.runtime_info)
```

Use `program.data.shape` and measurement metadata to understand the array;
do not assume all backends return the same dimensionality or axis ordering.
`program.counts` is `None` for non-STATE results. Although the SDK enum includes
`RAW`, the current cloud request validator accepts only `state` and `iq`.
Remote IQ classification centers are provided by the service-side configuration;
the client rejects caller-supplied `iq_centers` for remote execution.

## Batches and sweep coordinates

Every returned program has its own data and metadata. Read
`result.program_results[0].counts` for one STATE program; do not use the removed
top-level `result.counts` shortcut.

```python
# Given a completed batch result:
for program in result.program_results:
    print(program.index, program.name, program.coordinates, program.data)
```

The current SDK derives `program.coordinates` from `program.metadata["sweep"]`.
Legacy results containing only a separate `coordinates` field may not expose
their sweep coordinates through this property.

## Errors and recovery

| Symptom | What to check |
| --- | --- |
| Invalid or revoked credential | Use a valid API key from the target service |
| Resource unavailable or qubit rejected | Resource status, slug, and available physical indices |
| Job remains queued | Worker readiness, queue routing, and scheduler logs |
| Execution failure | `job.details()` and worker error; avoid blind resubmission |
| Wait timeout | Query the saved job ID; timeout does not cancel the job |
| HTTP timeout or connection failure | Base URL, network, TLS, and the service proxy |

See [Execution API](../api/execution.md) and [exception types](../api/protocol.md).
