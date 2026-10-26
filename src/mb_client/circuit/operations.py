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

"""Measurement, reset, delay, barrier, and drive instructions."""

from __future__ import annotations
from mb_client.circuit.circuit import Instruction


class Barrier(Instruction):

    def __init__(self, num_qubits: int):
        super().__init__("barrier", num_qubits)


VALID_CHANNELS = ("drive", "flux", "readout")
class Delay(Instruction):

    def __init__(self, duration: float, channel: str = "drive"):
        if isinstance(duration, (int, float)) and duration < 0:
            raise ValueError("Delay duration cannot be negative")
        if channel not in VALID_CHANNELS:
            raise ValueError(f"Invalid channel '{channel}', must be one of {VALID_CHANNELS}")
        self.duration = duration
        super().__init__("delay", 1, params=[self.duration, channel])


class Measure(Instruction):

    def __init__(self, **options):
        self.calibration = dict(options)
        super().__init__("measure", 1, 1)


class Reset(Instruction):

    def __init__(self, **calibration):
        self.calibration = dict(calibration)
        super().__init__("reset", 1)


class Drive(Instruction):
    """Parameterized low-level logical drive, calibrated by the backend."""

    def __init__(self, operation: str = "xy", **parameters):
        self.operation = operation
        self.parameters = dict(parameters)
        super().__init__("drive", 1, params=[operation, self.parameters])


class Idle(Delay):
    """Compatibility name for a circuit delay."""
