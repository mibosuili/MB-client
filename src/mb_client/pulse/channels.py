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

"""Pulse construction and scheduling primitives."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class _Channel:
    qubit_index: int

    def __post_init__(self):
        if type(self.qubit_index) is not int or self.qubit_index < 0:
            raise ValueError("channel index must be a non-negative integer")

    def __str__(self):
        return f"{type(self).__name__}({self.qubit_index})"


class DriveChannel(_Channel):
    """Qubit drive output."""


class FluxChannel(_Channel):
    """Qubit flux output."""


class ReadoutChannel(_Channel):
    """Qubit readout output and acquisition endpoints."""


Channel: TypeAlias = DriveChannel | FluxChannel | ReadoutChannel


def channel_name(channel: Channel, *, acquire=False, frequency=None) -> str:
    """Canonical endpoint name at the PulseCollection output boundary."""
    i = channel.qubit_index
    if isinstance(channel, ReadoutChannel):
        return f"r{i}.{'dig' if acquire else 'xy'}"
    return f"q{i}.{'xy' if isinstance(channel, DriveChannel) else 'z'}"


__all__ = [
    "Channel", "DriveChannel", "FluxChannel", "ReadoutChannel", "channel_name"
]
