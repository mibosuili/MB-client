# Copyright 2026 mibosuili
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned, JSON-safe circuit protocol shared by client and server."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .execution.results import ExecutionResult, ProgramResult
from .execution.options import MeasurementLevel, RunOptions

from .circuit import QuantumCircuit


SCHEMA = "mb-circuit"
VERSION = "1.0"
SUPPORTED_OPERATIONS = {
    "barrier", "cx", "cz", "delay", "h", "id", "measure", "reset",
    "rx", "ry", "rxy", "rz", "sx", "sy", "x", "y",
}


def encode_circuit(circuit: QuantumCircuit) -> dict[str, Any]:
    """Encode a circuit without serializing executable Python objects."""
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("program must be a QuantumCircuit")
    operations = []
    for item in circuit:
        operation = item.operation
        if operation.name not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported circuit operation: {operation.name}")
        params = [_json_number(value) for value in operation.params]
        operations.append({
            "name": operation.name,
            "qubits": [bit.index for bit in item.qubits],
            "clbits": [bit.index for bit in item.clbits],
            "params": params,
        })
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "name": circuit.name,
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "physical_qubits": list(circuit.physical_qubits),
        "operations": operations,
        "metadata": _json_mapping(circuit.metadata),
    }


def decode_circuit(payload: dict[str, Any]) -> QuantumCircuit:
    """Decode a validated protocol payload using an operation whitelist."""
    if not isinstance(payload, dict):
        raise TypeError("circuit payload must be an object")
    if payload.get("schema") != SCHEMA or payload.get("version") != VERSION:
        raise ValueError("Unsupported circuit schema or version")
    num_qubits = _positive_size(payload.get("num_qubits"), "num_qubits")
    num_clbits = _non_negative_size(payload.get("num_clbits"), "num_clbits")
    name = payload.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("circuit name must be a string")
    circuit = QuantumCircuit(num_qubits, num_clbits, name=name)
    physical = payload.get("physical_qubits", list(range(num_qubits)))
    if (not isinstance(physical, list) or len(physical) != num_qubits
            or any(type(value) is not int or value < 0 for value in physical)
            or len(set(physical)) != len(physical)):
        raise ValueError("physical_qubits must contain distinct non-negative integers")
    circuit.physical_qubits = tuple(physical)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("circuit metadata must be an object")
    circuit.metadata = metadata.copy()
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("circuit operations must be an array")
    for operation in operations:
        _append_operation(circuit, operation)
    return circuit


def _append_operation(circuit: QuantumCircuit, value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError("each circuit operation must be an object")
    name = value.get("name")
    if name not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported circuit operation: {name}")
    qubits = _indices(value.get("qubits", []), circuit.num_qubits, "qubits")
    clbits = _indices(value.get("clbits", []), circuit.num_clbits, "clbits")
    params = value.get("params", [])
    if not isinstance(params, list):
        raise ValueError("operation params must be an array")
    for param in params:
        _json_number(param)

    if name == "barrier":
        if not qubits or clbits or params:
            raise ValueError("barrier requires qubits and no parameters")
        circuit.barrier(*qubits)
    elif name == "measure":
        if len(qubits) != 1 or len(clbits) != 1:
            raise ValueError("measure requires one qubit and one classical bit")
        circuit.measure(qubits[0], clbits[0])
    elif name == "delay":
        if len(qubits) != 1 or clbits or len(params) != 2 or not isinstance(
                params[1], str):
            raise ValueError("delay requires duration and operation parameters")
        circuit.delay(params[0], qubits[0], params[1])
    elif name == "reset":
        if len(qubits) != 1 or clbits or params:
            raise ValueError("reset requires one qubit and no parameters")
        circuit.reset(*qubits, align=False)
    elif name in {"x", "y", "sx", "sy", "h"}:
        if len(qubits) != 1 or params:
            raise ValueError(f"{name} requires one qubit and no parameters")
        getattr(circuit, name)(qubits[0])
    elif name == "id":
        if len(qubits) != 1 or clbits or len(params) > 1:
            raise ValueError("id requires one qubit and an optional duration")
        circuit.id(None if not params else params[0], qubits[0])
    elif name in {"rx", "ry", "rz"}:
        if len(qubits) != 1 or len(params) != 1:
            raise ValueError(f"{name} requires one angle")
        getattr(circuit, name)(params[0], qubits[0])
    elif name == "rxy":
        if len(qubits) != 1 or len(params) != 2:
            raise ValueError("rxy requires theta and phi")
        circuit.rxy(params[0], params[1], qubits[0])
    elif name in {"cx", "cz"}:
        if len(qubits) != 2 or params:
            raise ValueError(f"{name} requires two qubits and no parameters")
        getattr(circuit, name)(*qubits)


def _indices(value: Any, size: int, name: str) -> list[int]:
    if (not isinstance(value, list)
            or any(type(index) is not int or not 0 <= index < size for index in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"operation {name} are invalid")
    return value


def _positive_size(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_size(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _json_number(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("circuit parameters must be finite numbers or strings")
    if not math.isfinite(value):
        raise ValueError("circuit parameters must be finite")
    return value


def _json_mapping(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    # Reuse the standard encoder as a strict JSON compatibility check.
    import json
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain JSON-compatible values") from exc


def encode_result(result: ExecutionResult) -> dict[str, Any]:
    """Encode execution results as JSON, including complex measurement arrays."""
    if not isinstance(result, ExecutionResult):
        raise TypeError("result must be an ExecutionResult")
    options = result.run_options
    return {
        "job_id": result.job_id,
        "backend_name": result.backend_name,
        "experiment_name": result.experiment_name,
        "metadata": _json_mapping(dict(result.metadata)),
        "run_options": {
            "shots": options.shots,
            "measure_level": options.measure_level.value,
            "iq_centers": None if options.iq_centers is None else {
                str(q): _encode_json_value(centers)
                for q, centers in options.iq_centers.items()
            },
        },
        "program_results": [{
            "index": item.index,
            "input_index": item.input_index,
            "program_type": item.program_type,
            "name": item.name,
            "physical_qubits": list(item.physical_qubits),
            "shots": item.shots,
            "measure_level": item.measure_level.value,
            "data": _encode_json_value(item.data),
            "metadata": _json_mapping(dict(item.metadata)),
            "runtime_info": _encode_json_value(item.runtime_info),
        } for item in result.program_results],
        "runtime_info": _encode_json_value(result.runtime_info),
    }


def _encode_json_value(value):
    if isinstance(value, np.ndarray):
        return {"real": _encode_json_value(value.real.tolist()),
                "imag": _encode_json_value(value.imag.tolist()),
                "shape": list(value.shape)}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return {"real": _encode_json_value(value.real),
                "imag": _encode_json_value(value.imag)}
    if isinstance(value, Mapping):
        return {key: _encode_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result values must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def decode_result(value: Mapping[str, Any]) -> ExecutionResult:
    """Reconstruct a unified execution result from the server JSON response."""
    if value.get("experiment") not in {None, "circuit"}:
        raise ValueError("The server returned an unsupported result type")
    options = value.get("run_options") or {}
    run_options = RunOptions(
        shots=options.get("shots", 1024),
        measure_level=options.get("measure_level", "state"),
        iq_centers=None if options.get("iq_centers") is None else {
            int(q): _decode_json_value(centers)
            for q, centers in options["iq_centers"].items()
        },
    )
    records = tuple(
        ProgramResult(
            index=item["index"],
            input_index=item.get("input_index", item["index"]),
            program_type=item["program_type"],
            name=item.get("name"),
            physical_qubits=tuple(item.get("physical_qubits", ())),
            shots=item.get("shots", run_options.shots),
            measure_level=MeasurementLevel(
                item.get("measure_level", run_options.measure_level.value)
            ),
            data=_decode_json_value(item.get("data")),
            metadata=item.get("metadata") or {},
            runtime_info=_decode_json_value(item.get("runtime_info") or {}),
        )
        for item in value.get("program_results", ())
    )
    return ExecutionResult(
        job_id=str(value["job_id"]),
        backend_name=str(value["backend_name"]),
        experiment_name=value.get("experiment_name", value.get("experiment")),
        metadata=value.get("metadata") or {},
        run_options=run_options,
        program_results=records,
        runtime_info=_decode_json_value(value.get("runtime_info") or {}),
    )


def _decode_json_value(value):
    if (isinstance(value, dict)
            and {"real", "imag", "shape"}.issubset(value)):
        data = np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
        return np.asarray(data).reshape(tuple(value["shape"]))
    if isinstance(value, dict) and set(value) == {"real", "imag"}:
        return complex(value["real"], value["imag"])
    if isinstance(value, dict):
        return {key: _decode_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    return value
