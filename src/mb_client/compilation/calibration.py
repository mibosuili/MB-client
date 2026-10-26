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

"""compilation: calibration APIs."""

from __future__ import annotations
import copy
from datetime import datetime, timezone
import pandas as pd


class CalibrationError(LookupError):
    pass


def _validate_qubits(qubits):
    if (not isinstance(qubits, tuple) or not qubits
            or any(type(q) is not int or q < 0 for q in qubits)
            or len(set(qubits)) != len(qubits)):
        raise ValueError("calibration qubit keys must be non-empty tuples of unique non-negative integers")


class CalibrationParams:
    """Calibration rows keyed by ordered physical-qubit tuples in ``qubit``."""

    REQUIRED_COLUMNS = ("qubit", "parameter", "value")
    OPTIONAL_COLUMNS = ("gate", "group", "timestamp", "metadata")
    COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

    def __init__(self, dataframe: pd.DataFrame | None = None):
        frame = (pd.DataFrame(columns=self.COLUMNS)
                 if dataframe is None else _deepcopy_frame(dataframe))
        missing = set(self.REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing calibration columns: {sorted(missing)}")
        for qubits in frame["qubit"]:
            _validate_qubits(qubits)
        for column in self.OPTIONAL_COLUMNS:
            if column not in frame:
                frame[column] = None
        frame["gate"] = frame["gate"].replace("", None)
        self._frame = frame.loc[:, self.COLUMNS].reset_index(drop=True)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> "CalibrationParams":
        return cls(dataframe)

    def to_dataframe(self) -> pd.DataFrame:
        return _deepcopy_frame(self._frame)

    def get(self, qubit: tuple[int, ...], parameter: str, *, gate: str | None = None):
        _validate_qubits(qubit)
        rows = self._frame[(self._frame.qubit.map(
            lambda value: value == qubit))
                           & (self._frame.parameter == parameter)]
        if gate is not None:
            exact = rows[rows.gate == gate]
            if not exact.empty:
                return copy.deepcopy(exact.iloc[-1].value)
        general = rows[rows.gate.isna() | (rows.gate == "")]
        if not general.empty:
            return copy.deepcopy(general.iloc[-1].value)
        raise CalibrationError(
            f"No calibration value for {qubit}.{parameter}" +
            (f" and gate {gate!r}" if gate is not None else ""))

    def set(self,
            *,
            qubit: tuple[int, ...],
            parameter: str,
            value,
            gate: str | None = None,
            group: str | None = None,
            metadata=None) -> "CalibrationParams":
        row = {
            "qubit": qubit,
            "parameter": parameter,
            "value": value,
            "gate": gate,
            "group": group,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata,
        }
        if self._frame.empty:
            frame = pd.DataFrame([row], columns=self.COLUMNS)
        else:
            frame = pd.concat([self._frame, pd.DataFrame([row])],
                              ignore_index=True)
        return CalibrationParams(frame)


def _deepcopy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for column in result.columns:
        result[column] = pd.Series(
            [copy.deepcopy(value) for value in frame[column]],
            index=frame.index,
            dtype=frame[column].dtype)
    return result
