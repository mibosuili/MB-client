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

"""Drawing backends for :class:`~mb_client.circuit.QuantumCircuit`."""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import html
from importlib import resources
import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping
from mb_client.common.parameters import Parameter, ParameterExpression

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mb_client.circuit.circuit import CircuitInstruction, QuantumCircuit

StyleInput = Mapping[str, Any] | str | PathLike[str] | None
DrawFormat = Literal["text", "mpl"]


class TextDrawing(str):
    """A string that displays as a multiline, monospace drawing in notebooks."""

    def __repr__(self) -> str:
        return str(self)

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        printer.text(str(self))

    def _repr_html_(self) -> str:
        return ("<pre style=\"font-family: monospace; line-height: 1.2\">" +
                html.escape(str(self)) + "</pre>")


@dataclass(frozen=True)
class _PlacedInstruction:
    instruction: "CircuitInstruction"
    column: int
    rows: tuple[int, ...]


class BaseDrawer(ABC):
    """Base class shared by circuit drawing backends."""

    def __init__(self,
                 circuit: "QuantumCircuit",
                 style: StyleInput = None) -> None:
        self.circuit = circuit
        self.style = self._load_style(style)
        self.layout = self._build_layout()

    @abstractmethod
    def draw(self) -> object:
        """Render the circuit in this drawer's output format."""

    @staticmethod
    def _deep_merge(base: dict[str, Any],
                    override: Mapping[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(base.get(key), dict):
                BaseDrawer._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base

    @classmethod
    def _load_style(cls, style: StyleInput) -> dict[str, Any]:
        default = resources.files(__package__).joinpath("styles/default.json")
        with default.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if style is None:
            override: Mapping[str, Any] = {}
        elif isinstance(style, Mapping):
            override = style
        elif isinstance(style, (str, PathLike)):
            try:
                with Path(style).open("r", encoding="utf-8") as stream:
                    override = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Cannot load circuit style {style!r}: {exc}") from exc
            if not isinstance(override, Mapping):
                raise ValueError("Circuit style JSON must contain an object")
        else:
            raise TypeError("style must be a mapping, JSON path, or None")
        cls._deep_merge(result, override)
        cls._validate_style(result)
        return result

    @staticmethod
    def _validate_style(style: Mapping[str, Any]) -> None:
        required = {
            "layout":
            ("column_width", "column_spacing", "row_spacing", "padding",
             "figure_column_width", "figure_column_spacing",
             "figure_marker_width", "figure_barrier_width",
             "figure_wire_spacing", "font_size", "subfont_size"),
            "text":
            ("quantum_wire", "classical_wire", "vertical_wire", "barrier",
             "control", "target", "measure", "measure_arrow", "box_left",
             "box_right", "multi_top_left", "multi_top_right",
             "multi_bottom_left", "multi_bottom_right", "multi_horizontal",
             "multi_vertical", "multi_wire_left", "multi_wire_right",
             "gate_top_left", "gate_top_right", "gate_bottom_left",
             "gate_bottom_right", "gate_horizontal", "gate_wire_left",
             "gate_wire_right", "shared_left", "shared_right",
             "measurement_vertical", "shared_from_above", "shared_from_below",
             "measurement_crossing", "measurement_gate_join",
             "classical_join"),
            "mpl": ("background_color", "wire_color", "classical_wire_color",
                    "connection_color", "label_color", "gate_facecolor",
                    "gate_edgecolor", "gate_textcolor", "barrier_color",
                    "measurement_facecolor", "line_width", "gate_height",
                    "minimum_gate_width", "character_width",
                    "classical_line_gap", "control_radius"),
            "gates": (),
        }
        for section, keys in required.items():
            value = style.get(section)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"Circuit style section {section!r} must be an object")
            missing = [key for key in keys if key not in value]
            if missing:
                raise ValueError(
                    f"Circuit style section {section!r} is missing {missing}")
        layout = style["layout"]
        for key in ("column_width", "figure_column_width",
                    "figure_column_spacing", "figure_marker_width",
                    "figure_barrier_width", "figure_wire_spacing", "font_size",
                    "subfont_size"):
            if not isinstance(layout[key], (int, float)) or layout[key] <= 0:
                raise ValueError(
                    f"Circuit style layout.{key} must be positive")
        for key in ("column_spacing", "row_spacing"):
            if not isinstance(layout[key], int) or layout[key] < 0:
                raise ValueError(
                    f"Circuit style layout.{key} must be a non-negative integer"
                )

    def _instruction_rows(self, item: "CircuitInstruction") -> tuple[int, ...]:
        rows = [bit.index for bit in item.qubits]
        rows.extend(self.circuit.num_qubits + bit.index for bit in item.clbits)
        if not rows:
            return ()
        # Reserve the entire visual span so connections never cross a peer gate.
        return tuple(range(min(rows), max(rows) + 1))

    def _build_layout(self) -> tuple[_PlacedInstruction, ...]:
        last_column = [-1
                       ] * (self.circuit.num_qubits + self.circuit.num_clbits)
        placed = []
        for item in self.circuit.data:
            rows = self._instruction_rows(item)
            column = max((last_column[row] + 1 for row in rows), default=0)
            for row in rows:
                last_column[row] = column
            placed.append(_PlacedInstruction(item, column, rows))
        return tuple(placed)

    @property
    def num_columns(self) -> int:
        return max((item.column for item in self.layout), default=-1) + 1

    def gate_label(self, item: "CircuitInstruction") -> str:
        operation = item.operation
        gate_style = self.style["gates"].get(operation.name, {})
        label = gate_style.get("label", operation.name)
        if operation.params:
            params = ",".join(
                self._format_value(value) for value in operation.params)
            label = f"{label}({params})"
        return str(label)

    def gate_style(self, name: str, key: str, default: Any) -> Any:
        return self.style["gates"].get(name, {}).get(key, default)

    def gate_title(self, item: "CircuitInstruction") -> str:
        operation = item.operation
        return str(self.style["gates"].get(operation.name,
                                           {}).get("label", operation.name))

    def parameter_label(self, item: "CircuitInstruction") -> str:
        return ", ".join(
            self._format_value(value) for value in item.operation.params)

    @classmethod
    def _format_value(cls, value: object) -> str:
        if isinstance(value, Parameter):
            return {"theta": "θ", "phi": "φ"}.get(value.name, value.name)
        if isinstance(value, ParameterExpression):
            left, *rest = value.operands
            rendered = cls._format_value(left)
            for operand in rest:
                rendered += value.symbol + cls._format_value(operand)
            return rendered
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, Mapping):
            return "{...}"
        return str(value)


class TextDrawer(BaseDrawer):
    """Render a circuit as a Unicode text diagram."""

    def draw(self) -> TextDrawing:
        wire_count = self.circuit.num_qubits + (1 if self.circuit.num_clbits
                                                else 0)
        if wire_count == 0:
            return TextDrawing("")
        text = self.style["text"]
        spacing = self._wire_spacing
        column_spacing = int(self.style["layout"]["column_spacing"])
        labels = [f"q_{index}: " for index in range(self.circuit.num_qubits)]
        if self.circuit.num_clbits:
            labels.append(f"c: {self.circuit.num_clbits}/")
        prefix = max(map(len, labels), default=0)
        widths = [1] * self.num_columns
        for placed in self.layout:
            item = placed.instruction
            name = item.operation.name
            if name == "measure":
                required = len(str(text["measure"])) + 2
            elif name == "cx":
                required = len(str(text["target"])) + 4
            elif name in ("barrier", "cz"):
                required = 1
            else:
                required = max(int(self.style["layout"]["column_width"]),
                               len(self.gate_label(item)) + 4)
            widths[placed.column] = max(widths[placed.column], required)
        centers = []
        cursor = prefix
        for width in widths:
            centers.append(cursor + width // 2)
            cursor += width + column_spacing
        canvas_width = max(cursor + 1, prefix + 2)
        canvas = [[" " for _ in range(canvas_width)]
                  for _ in range((wire_count - 1) * spacing + 4)]
        for row, label in enumerate(labels):
            y = row * spacing + 1
            padded = label.rjust(prefix)
            canvas[y][:prefix] = padded
            wire = (text["quantum_wire"] if row < self.circuit.num_qubits else
                    text["classical_wire"])
            for x in range(prefix, canvas_width):
                canvas[y][x] = wire

        for placed in self.layout:
            self._draw_instruction(canvas, centers[placed.column], placed)
        lines = ["".join(line).rstrip() for line in canvas]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return TextDrawing("\n".join(lines))

    def _put(self, canvas: list[list[str]], y: int, x: int,
             value: str) -> None:
        if 0 <= y < len(canvas) and 0 <= x < len(canvas[y]):
            existing = canvas[y][x]
            text = self.style["text"]
            if {existing, value
                } == {text["gate_bottom_left"], text["gate_top_left"]}:
                value = text["shared_left"]
            elif {existing, value
                  } == {text["gate_bottom_right"], text["gate_top_right"]}:
                value = text["shared_right"]
            elif (existing
                  in (text["gate_bottom_left"], text["gate_bottom_right"])
                  and value == text["gate_horizontal"]):
                value = text["shared_from_above"]
            elif (value in (text["gate_top_left"], text["gate_top_right"])
                  and existing == text["gate_horizontal"]):
                value = text["shared_from_below"]
            canvas[y][x] = value

    @property
    def _wire_spacing(self) -> int:
        # A three-line gate occupies the wire row and one line on either side.
        # Zero row spacing therefore places neighboring wire centers two lines apart.
        return 2 + int(self.style["layout"]["row_spacing"])

    def _box(self,
             canvas: list[list[str]],
             row: int,
             x: int,
             label: str,
             *,
             compact: bool = False) -> None:
        text = self.style["text"]
        width = len(label) + (2 if compact else 4)
        left = x - width // 2
        right = left + width - 1
        y = row * self._wire_spacing + 1
        top = (text["gate_top_left"] + text["gate_horizontal"] * (width - 2) +
               text["gate_top_right"])
        content = label if compact else f" {label} "
        middle = text["gate_wire_left"] + content + text["gate_wire_right"]
        bottom = (text["gate_bottom_left"] + text["gate_horizontal"] *
                  (width - 2) + text["gate_bottom_right"])
        for line_y, value in ((y - 1, top), (y, middle), (y + 1, bottom)):
            for offset, char in enumerate(value):
                self._put(canvas, line_y, left + offset, char)

    def _connect(self, canvas: list[list[str]], first: int, last: int,
                 x: int) -> None:
        spacing = self._wire_spacing
        for y in range(first * spacing + 2, last * spacing + 1):
            self._put(canvas, y, x, self.style["text"]["vertical_wire"])

    def _draw_instruction(self, canvas, x: int,
                          placed: _PlacedInstruction) -> None:
        item = placed.instruction
        name = item.operation.name
        qrows = [bit.index for bit in item.qubits]
        crows = [self.circuit.num_qubits] if item.clbits else []
        spacing = self._wire_spacing
        text = self.style["text"]
        if name == "barrier":
            top = min(qrows) * spacing
            bottom = max(qrows) * spacing + 2
            for y in range(top, bottom + 1):
                self._put(canvas, y, x, text["barrier"])
            return
        if name == "measure" and qrows and crows:
            self._box(canvas, qrows[0], x, str(text["measure"]), compact=True)
            quantum_y = qrows[0] * spacing + 1
            classical_y = crows[0] * spacing + 1
            self._put(canvas, quantum_y + 1, x, text["measurement_gate_join"])
            for y in range(quantum_y + 2, classical_y):
                char = (text["measurement_crossing"] if (y - 1) %
                        spacing == 0 else text["measurement_vertical"])
                self._put(canvas, y, x, char)
            self._put(canvas, classical_y, x, text["classical_join"])
            bit_index = str(item.clbits[0].index)
            for offset, char in enumerate(bit_index):
                self._put(canvas, classical_y + 1, x + offset, char)
            return
        if name in ("cx", "cz") and len(qrows) == 2:
            self._connect(canvas, min(qrows), max(qrows), x)
            self._put(canvas, qrows[0] * spacing + 1, x, text["control"])
            if name == "cx":
                self._box(canvas, qrows[1], x, str(text["target"]))
                self._put(canvas, qrows[1] * spacing, x, "┴")
            else:
                self._put(canvas, qrows[1] * spacing + 1, x, text["control"])
            return
        if len(qrows) > 1:
            self._multi_box(canvas, qrows, x, self.gate_label(item))
            return
        for row in qrows or crows:
            self._box(canvas, row, x, self.gate_label(item))

    def _multi_box(self, canvas: list[list[str]], rows: list[int], x: int,
                   label: str) -> None:
        """Draw one box spanning all operands of a multi-qubit instruction."""
        text = self.style["text"]
        spacing = self._wire_spacing
        first, last = min(rows), max(rows)
        width = max(9, len(label) + 4)
        left = x - width // 2
        right = left + width - 1
        top = first * spacing
        bottom = last * spacing + 2
        for column in range(left + 1, right):
            self._put(canvas, top, column, text["multi_horizontal"])
            self._put(canvas, bottom, column, text["multi_horizontal"])
        self._put(canvas, top, left, text["multi_top_left"])
        self._put(canvas, top, right, text["multi_top_right"])
        self._put(canvas, bottom, left, text["multi_bottom_left"])
        self._put(canvas, bottom, right, text["multi_bottom_right"])
        for y in range(top + 1, bottom):
            for column in range(left + 1, right):
                self._put(canvas, y, column, " ")
            self._put(canvas, y, left, text["multi_vertical"])
            self._put(canvas, y, right, text["multi_vertical"])
        for row in rows:
            y = row * spacing + 1
            self._put(canvas, y, left, text["multi_wire_left"])
            self._put(canvas, y, right, text["multi_wire_right"])
        label_y = (top + bottom) // 2
        label_start = x - len(label) // 2
        for offset, char in enumerate(label):
            self._put(canvas, label_y, label_start + offset, char)


def draw_circuit(circuit: "QuantumCircuit",
                 format: str = "text",
                 style: StyleInput = None) -> str | "Figure":
    """Select a drawing backend and render ``circuit``."""
    if not isinstance(format, str):
        raise TypeError("format must be a string")
    normalized = format.lower()
    from .matplotlib import MatplotlibDrawer
    drawers = {"text": TextDrawer, "mpl": MatplotlibDrawer}
    try:
        drawer = drawers[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unsupported circuit drawing format; use 'text' or 'mpl'") from exc
    return drawer(circuit, style).draw()
