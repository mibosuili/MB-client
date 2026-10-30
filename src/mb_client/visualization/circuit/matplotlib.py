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

"""visualization: circuit: matplotlib APIs."""

from __future__ import annotations
from .drawer import BaseDrawer


class MatplotlibDrawer(BaseDrawer):
    """Render a circuit as a Matplotlib figure."""

    def draw(self) -> "Figure":
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

        mpl = self.style["mpl"]
        layout = self.style["layout"]
        count = self.circuit.num_qubits + (1 if self.circuit.num_clbits else 0)
        row_spacing = float(layout["figure_wire_spacing"])
        column_widths = self._column_widths()
        column_spacing = float(layout["figure_column_spacing"])
        centers = []
        cursor = .2
        for column_width in column_widths:
            centers.append(cursor + column_width / 2)
            cursor += column_width + column_spacing
        x_end = max(cursor + .12, 1.0)
        width = max(3.0, x_end * 1.05)
        height = max(1.4, (max(count, 1) + .7) * row_spacing)
        fig = Figure(figsize=(width, height),
                     facecolor=mpl["background_color"])
        ax = fig.subplots()
        ax.set_facecolor(mpl["background_color"])
        ys = [(count - 1 - row) * row_spacing for row in range(count)]
        for row, y in enumerate(ys):
            quantum = row < self.circuit.num_qubits
            color = mpl["wire_color"] if quantum else mpl[
                "classical_wire_color"]
            offsets = (0, ) if quantum else (-mpl["classical_line_gap"],
                                             mpl["classical_line_gap"])
            for offset in offsets:
                ax.plot([0, x_end], [y + offset, y + offset],
                        color=color,
                        linewidth=mpl["line_width"],
                        zorder=0)
            if row < self.circuit.num_qubits:
                label = f"q_{row}:"
            else:
                label = f"c: {self.circuit.num_clbits}/"
            ax.text(-0.08,
                    y,
                    label,
                    ha="right",
                    va="center",
                    color=mpl["label_color"],
                    fontsize=layout["font_size"])
        for placed in self.layout:
            x = centers[placed.column]
            item = placed.instruction
            name = item.operation.name
            qrows = [bit.index for bit in item.qubits]
            crows = [self.circuit.num_qubits] if item.clbits else []
            rows = qrows + crows
            if name == "barrier":
                top, bottom = ys[min(qrows)] + .27, ys[max(qrows)] - .27
                ax.axvspan(x - .035,
                           x + .035,
                           ymin=(bottom - (ys[-1] - row_spacing)) /
                           ((ys[0] + row_spacing) - (ys[-1] - row_spacing)),
                           ymax=(top - (ys[-1] - row_spacing)) /
                           ((ys[0] + row_spacing) - (ys[-1] - row_spacing)),
                           facecolor=mpl["barrier_color"],
                           alpha=.55,
                           hatch="//",
                           edgecolor=mpl["barrier_color"],
                           linewidth=0)
                continue
            if len(rows) > 1 and name != "measure":
                ax.plot([x, x], [ys[max(rows)], ys[min(rows)]],
                        color=mpl["connection_color"],
                        linewidth=mpl["line_width"],
                        zorder=1)
            if name in ("cx", "cz") and len(qrows) == 2:
                control_radius = float(mpl["control_radius"])
                ax.add_patch(
                    Circle((x, ys[qrows[0]]),
                           control_radius,
                           color=mpl["connection_color"],
                           zorder=3))
                if name == "cz":
                    ax.add_patch(
                        Circle((x, ys[qrows[1]]),
                               control_radius,
                               color=mpl["connection_color"],
                               zorder=3))
                else:
                    ax.add_patch(
                        Circle((x, ys[qrows[1]]),
                               .145,
                               fill=False,
                               color=mpl["connection_color"],
                               zorder=3))
                    ax.plot([x - .145, x + .145], [ys[qrows[1]], ys[qrows[1]]],
                            color=mpl["connection_color"],
                            zorder=3)
                    ax.plot([x, x], [ys[qrows[1]] - .145, ys[qrows[1]] + .145],
                            color=mpl["connection_color"],
                            zorder=3)
                continue
            label = self.gate_title(item)
            params = self.parameter_label(item)
            gate_width = self._gate_width(item)
            face = (mpl["measurement_facecolor"] if name == "measure" else
                    self.gate_style(name, "facecolor", mpl["gate_facecolor"]))
            edge = self.gate_style(name, "edgecolor", mpl["gate_edgecolor"])
            textcolor = self.gate_style(name, "textcolor",
                                        mpl["gate_textcolor"])
            target_rows = qrows or crows
            top_row, bottom_row = min(target_rows), max(target_rows)
            gate_height = (ys[top_row] - ys[bottom_row] +
                           float(mpl["gate_height"]))
            center_y = (ys[top_row] + ys[bottom_row]) / 2
            patch = FancyBboxPatch(
                (x - gate_width / 2, center_y - gate_height / 2),
                gate_width,
                gate_height,
                boxstyle="round,pad=0,rounding_size=0.025",
                facecolor=face,
                edgecolor=edge,
                linewidth=mpl["line_width"],
                zorder=2)
            ax.add_patch(patch)
            if name == "measure":
                ax.text(x,
                        center_y,
                        label,
                        ha="center",
                        va="center",
                        color=textcolor,
                        fontsize=layout["font_size"],
                        zorder=3)
            else:
                text_y = center_y + (.055 if params else 0)
                ax.text(x,
                        text_y,
                        label,
                        ha="center",
                        va="center",
                        color=textcolor,
                        fontsize=layout["font_size"],
                        zorder=3)
                if params:
                    ax.text(x,
                            center_y - .105,
                            params,
                            ha="center",
                            va="center",
                            color=textcolor,
                            fontsize=layout["subfont_size"],
                            zorder=3)
            if name == "measure" and crows:
                gate_bottom = center_y - gate_height / 2
                arrow = FancyArrowPatch(
                    (x, gate_bottom),
                    (x, ys[crows[0]] + float(mpl["classical_line_gap"])),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    shrinkA=0,
                    shrinkB=0,
                    color=mpl["connection_color"],
                    zorder=3)
                ax.add_patch(arrow)
                ax.text(x,
                        ys[crows[0]] - .16,
                        str(item.clbits[0].index),
                        ha="center",
                        va="top",
                        color=mpl["label_color"],
                        fontsize=layout["subfont_size"])
        ax.set_xlim(-.65, x_end + .08)
        if count:
            ax.set_ylim(ys[-1] - row_spacing, ys[0] + row_spacing)
        else:
            ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.tight_layout()
        return fig

    def _gate_width(self, item: "CircuitInstruction") -> float:
        mpl = self.style["mpl"]
        longest = max(len(self.gate_title(item)),
                      len(self.parameter_label(item)))
        return max(float(mpl["minimum_gate_width"]),
                   .18 + longest * float(mpl["character_width"]))

    def _column_widths(self) -> list[float]:
        minimum = float(self.style["layout"]["figure_column_width"])
        marker = float(self.style["layout"]["figure_marker_width"])
        barrier = float(self.style["layout"]["figure_barrier_width"])
        widths = [marker] * self.num_columns
        for placed in self.layout:
            name = placed.instruction.operation.name
            if name == "barrier":
                needed = barrier
            elif name == "cz":
                needed = marker
            elif name == "cx":
                needed = max(marker, self._gate_width_for_label("X"))
            else:
                needed = self._gate_width(placed.instruction)
            needed = max(needed, minimum) if name not in ("barrier",
                                                          "cz") else needed
            widths[placed.column] = max(widths[placed.column], needed)
        return widths

    def _gate_width_for_label(self, label: str) -> float:
        mpl = self.style["mpl"]
        return max(float(mpl["minimum_gate_width"]),
                   .18 + len(label) * float(mpl["character_width"]))
