# MBCloudClient

The HTTPS adapter for MB Cloud. Connection settings, resource queries, and
job submission use this class. Constructing a client does not submit a job;
methods such as `run()` do.

See [configuration](../getting-started/configuration.md) and the
[complete remote example](../examples/production-simulator.md).

::: mb_client.client.MBCloudClient
    options:
      members:
        - ping
        - resources
        - profile
        - backend
        - run
        - submit
        - status
        - details
        - results
        - cancel
