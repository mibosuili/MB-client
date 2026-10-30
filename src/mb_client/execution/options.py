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

"""Measurement levels and validated execution options."""

from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence
import numpy as np

IQCenters = Mapping[int, Sequence[complex]]


class MeasurementLevel(Enum):
    RAW = "raw"
    IQ = "iq"
    STATE = "state"


@dataclass(frozen=True)
class RunOptions:
    shots: int = 1024
    measure_level: MeasurementLevel = MeasurementLevel.STATE
    iq_centers: IQCenters | None = None

    def __post_init__(self):
        if type(self.shots) is not int or self.shots <= 0:
            raise ValueError("shots must be a positive integer")
        object.__setattr__(self, "measure_level",
                           MeasurementLevel(self.measure_level))
        if self.iq_centers is not None:
            if not isinstance(self.iq_centers, Mapping):
                raise TypeError(
                    "iq_centers must map physical qubits to center sequences")
            centers = {}
            for q, values in self.iq_centers.items():
                if type(q) is not int or q < 0:
                    raise ValueError(
                        "iq_centers keys must be non-negative physical qubits")
                array = np.asarray(values, dtype=complex)
                if array.ndim != 1 or not len(array) or not np.isfinite(
                        array).all():
                    raise ValueError(
                        "IQ centers must be nonempty finite complex sequences")
                centers[q] = tuple(complex(v) for v in array)
            object.__setattr__(self, "iq_centers", MappingProxyType(centers))

    def with_overrides(self, **overrides) -> RunOptions:
        unknown = set(overrides) - {"shots", "measure_level", "iq_centers"}
        if unknown:
            raise TypeError(
                f"Unknown run option(s): {', '.join(sorted(unknown))}. "
                "Allowed: shots, measure_level, iq_centers.")
        return replace(self, **overrides)
