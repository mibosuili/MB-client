# Serialization and errors

The circuit wire format is JSON with schema `mb-circuit`, version `1.0`. It does
not transport executable Python objects. The current encoders use an operation
whitelist; representing an operation locally does not guarantee it is encodable.

Result encoding supports complex values and arrays. `decode_result()` reconstructs
the public result model from the service response.

::: mb_client.protocol.encode_circuit

::: mb_client.protocol.decode_circuit

::: mb_client.protocol.encode_result

::: mb_client.protocol.decode_result

## Exceptions

::: mb_client.exceptions.ClientError

::: mb_client.exceptions.ServerConnectionError

::: mb_client.execution.jobs.JobExecutionError

::: mb_client.execution.jobs.JobCancelledError
