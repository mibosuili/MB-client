import json

import matplotlib.pyplot as plt
import pytest

from mb_client.common import Parameter
from mb_client.circuit import QuantumCircuit
from mb_client.visualization.circuit import (
    BaseDrawer,
    MatplotlibDrawer,
    TextDrawer,
    TextDrawing,
)


def test_text_drawer_and_string_protocol():
    theta = Parameter("theta")
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.rx(theta, 1)
    circuit.cx(0, 1)
    circuit.measure(1, 0)

    drawing = circuit.draw()

    assert drawing == str(circuit) == TextDrawer(circuit).draw()
    assert isinstance(drawing, TextDrawing)
    assert "\\n" not in repr(drawing)
    assert "<pre" in drawing._repr_html_()
    assert "┤ H ├" in drawing
    assert "┤ Rx(θ) ├" in drawing
    assert "■" in drawing and "┤ X ├" in drawing
    assert "q_0:" in drawing and "c: 1/" in drawing
    assert not drawing.endswith("\n")


def test_disjoint_instructions_share_a_compact_column():
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.y(1)

    drawer = TextDrawer(circuit)

    assert drawer.num_columns == 1
    lines = drawer.draw().splitlines()
    x_line = next(line for line in lines if "┤ X ├" in line)
    y_line = next(line for line in lines if "┤ Y ├" in line)
    assert x_line.index("┤ X ├") == y_line.index("┤ Y ├")
    assert "├───┤" in drawer.draw()


def test_text_row_and_column_spacing_are_styled():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.x(0)

    assert "┤ H ├─┤ X ├" in circuit.draw()
    assert "┤ H ├───┤ X ├" in circuit.draw(
        style={"layout": {"column_spacing": 3}})

    parallel = QuantumCircuit(2)
    parallel.h(0)
    parallel.x(1)
    assert "├───┤" in parallel.draw(style={"layout": {"row_spacing": 0}})


def test_text_shared_borders_barrier_and_measurement_junctions():
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.rx(3.14, 1)
    circuit.barrier()
    circuit.measure(0, 0)

    lines = circuit.draw().splitlines()

    assert any("┌───┴───┴" in line for line in lines)
    barrier_column = next(line.index("░") for line in lines if "░" in line)
    assert all(line[barrier_column] == "░" for line in lines[:5])
    assert "└╥┘" in "\n".join(lines)


def test_barrier_measurement_and_custom_gate_are_rendered():
    from mb_client.circuit import Gate

    circuit = QuantumCircuit(2, 1)
    circuit.append(Gate("custom", 2, [0.5]), [0, 1])
    circuit.barrier()
    circuit.measure(0, 0)

    drawing = circuit.draw()

    assert drawing.count("custom(0.5)") == 1
    assert "┌" in drawing and "┤             ├" in drawing
    assert "░" in drawing
    assert "┤M├" in drawing and "╩" in drawing


def test_style_mapping_and_json_path_are_isolated(tmp_path):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    assert "┤ NOT ├" in circuit.draw(style={"gates": {"x": {"label": "NOT"}}})

    style_path = tmp_path / "style.json"
    style_path.write_text(json.dumps({"text": {"gate_wire_left": "<",
                                                "gate_wire_right": ">"}}))
    assert "< X >" in circuit.draw(style=style_path)
    assert "┤ X ├" in circuit.draw()


def test_matplotlib_drawer_returns_figure_without_showing(monkeypatch):
    circuit = QuantumCircuit(2, 1)
    circuit.cx(0, 1)
    circuit.measure(1, 0)
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("show must not be called"))

    existing_figures = tuple(plt.get_fignums())
    figure = circuit.draw(format="MPL")

    assert isinstance(figure, plt.Figure)
    assert tuple(plt.get_fignums()) == existing_figures
    assert isinstance(MatplotlibDrawer(circuit).draw(), plt.Figure)
    assert len(figure.axes[0].lines) >= 3
    labels = [artist.get_text() for artist in figure.axes[0].texts]
    assert "M" in labels
    assert "c: 1/" in labels
    assert "0" in labels
    plt.close("all")


def test_draw_show_displays_text_and_matplotlib(monkeypatch):
    circuit = QuantumCircuit(1)
    circuit.x(0)
    displayed = []

    monkeypatch.setattr("IPython.display.display", displayed.append)

    text = circuit.draw(show=True)
    figure = circuit.draw(format="mpl", show=True)

    assert displayed == [text, figure]
    plt.close(figure)


def test_drawer_interface_and_invalid_inputs(tmp_path):
    assert BaseDrawer.__abstractmethods__ == {"draw"}
    circuit = QuantumCircuit(1)
    with pytest.raises(ValueError, match="Unsupported"):
        circuit.draw(format="svg")
    with pytest.raises(TypeError, match="style"):
        circuit.draw(style=object())
    bad = tmp_path / "bad.json"
    bad.write_text("[")
    with pytest.raises(ValueError, match="Cannot load"):
        circuit.draw(style=bad)
