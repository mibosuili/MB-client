# Installation

## Python requirements

The SDK supports **Python 3.10, 3.11, and 3.12** (`>=3.10,<3.13`). Use a virtual
environment to avoid conflicts with server, simulator, or hardware dependencies.

=== "macOS / Linux"

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows PowerShell"

    ```powershell
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

## Install from PyPI

```bash
python -m pip install --upgrade mibo-client
```

The public package is [mibo-client on PyPI](https://pypi.org/project/mibo-client/).
Use [the GitHub repository](https://github.com/mibosuili/MB-client) for source code
and notebook examples.

## Install a local checkout

Clone the public repository and install from its root, which contains `pyproject.toml`:

```bash
git clone https://github.com/mibosuili/MB-client.git
cd MB-client
python -m pip install -e .
```

## Verify the import

```bash
python -c "from mb_client import MBCloudClient, QuantumCircuit; print('Client import OK')"
```

The public distribution name is `mibo-client`; the Python import is `mb_client`.
The source repository is named `MB-client`. These names are intentionally different.

!!! tip "No hardware-side dependencies"
    Client users do not need instrument drivers, simulator packages, CUDA, Django, Redis, or MongoDB.
    GPU requirements apply to the simulator worker, not the client computer.

Next: [build a circuit](quickstart.md) or [configure a connection](configuration.md).
