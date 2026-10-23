# Connection and authentication

## Explicit configuration

```python
from mb_client import MBCloudClient

client = MBCloudClient(
    base_url="https://cloud.mbslabs.tech",
    api_key="YOUR_PRODUCTION_API_KEY",
    timeout=30.0,
)
```

`timeout` is the timeout for an individual HTTP request. The timeout passed to
`job.wait()` instead limits how long the client waits for a completed job.

The base URL is the service origin, **without** `/api/v1`; the client appends
the appropriate API route. Production clients use HTTPS, not a Redis URL.

## Environment variables and .env

The client uses Dynaconf with the `MB_CLOUD` environment prefix. A `.env` in the
working directory can contain:

```dotenv
MB_CLOUD_BASE_URL=https://cloud.mbslabs.tech
MB_CLOUD_API_KEY=YOUR_PRODUCTION_API_KEY
```

Then construct the client without arguments:

```python
from mb_client import MBCloudClient

client = MBCloudClient()
print(client.base_url)
```

Explicit nonempty constructor values override environment configuration.
Without a configured URL, the default is `http://127.0.0.1:8000`.
The client loads configuration at import time; configure your environment before
starting Python, or restart the process after changing `.env`.

!!! tip "Working directory matters"
    Start the script from the directory containing your `.env`. Merely placing
    it next to an arbitrary script does not guarantee it will be discovered.
    Explicit constructor values are the clearest option for a standalone test.

## API key handling

The client sends the key using the `X-API-Key` header. Create and revoke keys
through the web console for the service you are using.

- Never commit real credentials to Git, notebooks, screenshots, or public docs.
- Use separate keys for development and production.
- Revoke exposed keys and replace them in deployed applications.
- Keep TLS verification enabled; do not disable certificate checks to bypass an error.

For connection errors, see [Jobs and measurement results](../guides/jobs-and-results.md#errors-and-recovery).
