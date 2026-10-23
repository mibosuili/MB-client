# QuantumCircuit and parameters

Circuit construction is local. Resource mapping, protocol restrictions, and
parameter binding are explained in [Circuits and qubits](../guides/circuits.md).
Names below are resolved from the SDK source, not a separate hand-written schema.

::: mb_client.circuit.circuit.QuantumCircuit
    options:
      members:
        - num_qubits
        - num_clbits
        - parameters
        - bind_parameters
        - append
        - compose
        - to_gate
        - x
        - y
        - h
        - rx
        - ry
        - rz
        - rxy
        - cx
        - cz
        - id
        - delay
        - barrier
        - reset
        - measure
        - measure_all
        - draw

::: mb_client.common.parameters.Parameter

::: mb_client.circuit.circuit.Gate
