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

"""Symbolic parameters for logical circuits and pulse compilation."""

from __future__ import annotations
from dataclasses import dataclass, field
import operator
from typing import Mapping

_GROUPS = frozenset({"calibration", "operation"})


class _Unset:

    def __deepcopy__(self, memo):
        return self


UNSET = _Unset()


@dataclass(frozen=True)
class Parameter:
    """A symbolic operation value or calibration reference.

    Calibration ``qubits`` selects explicit physical qubits, preserving order.
    When omitted, the full qubit tuple of the gate registration is used.
    """

    name: str
    group: str = "operation"
    qubits: tuple[int, ...] | None = field(default=None, kw_only=True)
    gate: str | None = field(default=None, kw_only=True)
    default: object = field(default=UNSET,
                            compare=False,
                            hash=False,
                            kw_only=True)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Parameter name cannot be empty")
        if self.group not in _GROUPS:
            raise ValueError(f"Unknown parameter group {self.group!r}")
        if self.group == "operation" and (self.qubits is not None
                                          or self.gate is not None
                                          or self.has_default):
            raise ValueError(
                "operation parameters cannot have calibration selectors/defaults"
            )
        if self.qubits is not None:
            if (not isinstance(self.qubits, tuple) or not self.qubits
                    or any(type(q) is not int or q < 0 for q in self.qubits)
                    or len(set(self.qubits)) != len(self.qubits)):
                raise ValueError("qubits must be a non-empty tuple of unique non-negative integers")
        for value in (self.gate,):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("gate must be non-empty strings")

    @property
    def has_default(self):
        return self.default is not UNSET

    def __str__(self):
        return self.name if self.group == "operation" else f"cal({self.name})"

    def __neg__(self):
        return ParameterExpression(operator.neg, (self, ), "-")

    def bind(self, values: Mapping[object, object]):
        if self in values:
            return values[self]
        if self.group == "operation" and self.name in values:
            return values[self.name]
        return self

    def _binary(self, other, function, symbol):
        return ParameterExpression(function, (self, other), symbol)

    def __add__(self, other):
        return self._binary(other, operator.add, "+")

    def __sub__(self, other):
        return self._binary(other, operator.sub, "-")

    def __mul__(self, other):
        return self._binary(other, operator.mul, "*")

    def __truediv__(self, other):
        return self._binary(other, operator.truediv, "/")

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return ParameterExpression(operator.sub, (other, self), "-")

    def __rtruediv__(self, other):
        return ParameterExpression(operator.truediv, (other, self), "/")


@dataclass(frozen=True)
class ParameterExpression:
    function: object
    operands: tuple[object, ...]
    symbol: str

    def __neg__(self):
        return ParameterExpression(operator.neg, (self, ), "-")

    def __str__(self):
        if self.symbol not in ("+", "-", "*", "/"):
            return f"{self.symbol}({', '.join(map(str, self.operands))})"
        if len(self.operands) == 1:
            return f"(-{self.operands[0]})"
        return f"({self.operands[0]} {self.symbol} {self.operands[1]})"

    @property
    def parameters(self) -> frozenset[Parameter]:
        found = set()
        for operand in self.operands:
            found.update(parameters_in(operand))
        return frozenset(found)

    def bind(self, values: Mapping[object, object]):
        operands = tuple(bind_value(value, values) for value in self.operands)
        if any(parameters_in(value) for value in operands):
            return ParameterExpression(self.function, operands, self.symbol)
        return self.function(*operands)

    def _binary(self, other, function, symbol):
        return ParameterExpression(function, (self, other), symbol)

    __add__ = lambda self, other: self._binary(other, operator.add, "+")
    __sub__ = lambda self, other: self._binary(other, operator.sub, "-")
    __mul__ = lambda self, other: self._binary(other, operator.mul, "*")
    __truediv__ = lambda self, other: self._binary(other, operator.truediv, "/"
                                                   )
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = lambda self, other: ParameterExpression(operator.sub,
                                                       (other, self), "-")
    __rtruediv__ = lambda self, other: ParameterExpression(
        operator.truediv, (other, self), "/")


def parameters_in(value) -> frozenset[Parameter]:
    if isinstance(value, Parameter):
        return frozenset({value})
    if isinstance(value, ParameterExpression):
        return value.parameters
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return frozenset()
    return frozenset(parameter for item in values
                     for parameter in parameters_in(item))


def bind_value(value, values: Mapping[object, object]):
    if isinstance(value, (Parameter, ParameterExpression)):
        return value.bind(values)
    if isinstance(value, dict):
        return {key: bind_value(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [bind_value(item, values) for item in value]
    if isinstance(value, tuple):
        return tuple(bind_value(item, values) for item in value)
    return value
