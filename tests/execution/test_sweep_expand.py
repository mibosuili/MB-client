import pytest
from mb_client.circuit import QuantumCircuit
from mb_client.common import Parameter
from mb_client.execution import Sweep, SweepAxis
from mb_client.execution.sweep import BatchPreparation


@pytest.mark.parametrize('mode, count, shape', [('grid', 4, (2, 2, 2)), ('zip', 2, (2, 2))])
def test_public_expand_preserves_order_and_templates(mode, count, shape):
    a, b = Parameter('a'), Parameter('b')
    circuit = QuantumCircuit(1)
    circuit.rx(a, 0)
    circuit.ry(b, 0)
    sweep = getattr(Sweep, mode)(SweepAxis(a, [1, 2]), SweepAxis(b, [3, 4]))
    circuits = sweep.expand([circuit, circuit])
    rows = [program.metadata["sweep"] for program in circuits]
    assert len(circuits) == count * 2
    assert [(r['a'], r['b']) for r in rows[::2]] == (
        [(1, 3), (1, 4), (2, 3), (2, 4)] if mode == 'grid' else [(1, 3), (2, 4)])
    for bound, row in zip(circuits, rows):
        assert bound.data[0].operation.params == [row['a']]
        assert bound.data[1].operation.params == [row['b']]
        assert not bound.parameters
    assert circuit.parameters == {a, b}


def test_single_empty_and_unswept_inputs():
    p = Parameter('theta')
    circuit = QuantumCircuit(1)
    circuit.rx(p, 0)
    sweep = Sweep.grid(SweepAxis(p, [1, 2]))
    assert len(sweep.expand(circuit)) == 2
    assert sweep.expand([]) == []
    programs = BatchPreparation().expand([circuit], None)
    assert programs[0].metadata == circuit.metadata
    with pytest.raises(TypeError, match='QuantumCircuit'):
        sweep.expand([object()])
    with pytest.raises(ValueError, match='not in circuit'):
        sweep.expand(QuantumCircuit(1))


def test_sweep_metadata_survives_compilation_and_is_independent():
    from mb_client.circuit import XGate
    from mb_client.compilation import InstSchedRegistry, Target, schedule, lower
    from mb_client.pulse import PulseSequence, DriveChannel, Pulse, Constant
    amplitude = Parameter('sweep_amp', group='calibration')
    circuit = QuantumCircuit(1)
    circuit.metadata = {'experiment': {'labels': ['original']}}
    circuit.x(0, amplitude=amplitude)
    template = PulseSequence(template=True)
    template.play(DriveChannel(0), Pulse(Constant(
        20e-9, Parameter('amplitude', group='calibration'))))
    registry = InstSchedRegistry([0])
    registry.add(XGate(), (0,), template)
    target = Target(inst_sched_reg=registry)
    expanded = Sweep.grid(SweepAxis(amplitude, [.1, .2])).expand(circuit)
    assert expanded[0].metadata['sweep'] == {
        'sweep_amp': .1}
    scheduled = schedule(expanded, target=target)
    assert [p.metadata for p in scheduled] == [c.metadata for c in expanded]
    single = schedule(expanded[0], target=target)
    assert single.metadata == expanded[0].metadata
    single.metadata['experiment']['labels'].append('single')
    assert expanded[0].metadata['experiment']['labels'] == ['original']
    assert lower(expanded[0], target=target).schedule().metadata == expanded[0].metadata
    scheduled[0].metadata['experiment']['labels'].append('changed')
    assert circuit.metadata == {'experiment': {'labels': ['original']}}
    assert expanded[0].metadata['experiment']['labels'] == ['original']
    assert scheduled[1].metadata['experiment']['labels'] == ['original']
    assert scheduled[1].bind_parameters({}).metadata == scheduled[1].metadata


def test_sweep_values_are_read_only_and_template_values_are_not_overwritten():
    theta = Parameter('theta')
    circuit = QuantumCircuit(1)
    circuit.rx(theta, 0)
    circuit.metadata = {'note': {'labels': ['template']}}
    sweep = Sweep.grid(SweepAxis(theta, [1, 2]))
    expanded = sweep.expand(circuit)
    assert expanded[0].metadata['sweep'] == {'theta': 1}
    with pytest.raises(TypeError, match='cannot be updated'):
        expanded[0].metadata['sweep']['theta'] = 3
    with pytest.raises(TypeError, match='cannot be updated'):
        expanded[0].metadata['sweep'].update(theta=3)
    expanded[0].metadata['note']['labels'].append('edited')
    assert circuit.metadata == {'note': {'labels': ['template']}}
    assert expanded[1].metadata['note']['labels'] == ['template']
    circuit.metadata['sweep'] = {'theta': 0}
    with pytest.raises(ValueError, match='cannot overwrite'):
        sweep.expand(circuit)
