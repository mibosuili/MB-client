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

from collections import Counter
from collections.abc import Callable
import numpy as np


def classify_iq(
    iq_data,
    discriminator: Callable[[np.ndarray], str],
) -> dict[str, int]:
    """Classify IQ data shaped (shots, qubits); column 0 is rightmost."""
    iq_data = np.asarray(iq_data, dtype=complex)
    if iq_data.ndim != 2 or iq_data.shape[1] == 0:
        raise ValueError(
            "iq_data must have shape (shots, qubits) with at least one qubit"
        )
    if not np.isfinite(iq_data).all():
        raise ValueError("iq_data must contain finite values")
    if not callable(discriminator):
        raise TypeError("discriminator must be callable")

    counts = Counter()
    for shot in iq_data:
        state = discriminator(shot.copy())
        if not isinstance(state, str) or len(state) != iq_data.shape[1]:
            raise ValueError(
                "discriminator must return one state character per qubit"
            )
        counts[state] += 1
    return dict(counts)


def nearest_center_discriminator(
    state_centers,
) -> Callable[[np.ndarray], str]:
    """Centers have shape (qubits, 2): [state-0 center, state-1 center]."""
    centers = np.array(state_centers, dtype=complex, copy=True)
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("state_centers must have shape (qubits, 2)")
    if not np.isfinite(centers).all():
        raise ValueError("state_centers must contain finite values")

    def discriminate(iq_array: np.ndarray) -> str:
        iq_array = np.asarray(iq_array, dtype=complex)
        if iq_array.shape != (len(centers),) or not np.isfinite(iq_array).all():
            raise ValueError("iq_array must contain one finite value per qubit")

        states = np.argmin(np.abs(iq_array[:, None] - centers), axis=1)
        return "".join(str(state) for state in states[::-1])

    return discriminate