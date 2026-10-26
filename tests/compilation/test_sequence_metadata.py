from mb_client.circuit import QuantumCircuit, XGate
from mb_client.compilation import InstSchedRegistry, Target, lower, schedule
from mb_client.pulse import PulseSequence, Pulse, Constant, DriveChannel
from mb_client.pulse.sequence import transform_sequence
from mb_client.pulse.block import PulseBlock


def test_sequence_metadata_is_copied_through_transform_and_schedule():
    metadata = {'labels': ['original']}
    sequence = PulseSequence(metadata=metadata)
    sequence.play(DriveChannel(0), Pulse(Constant(20e-9, .2)))
    transformed = transform_sequence(sequence)
    scheduled = transformed.schedule()
    metadata['labels'].append('caller')
    sequence.metadata['labels'].append('sequence')
    transformed.metadata['labels'].append('transformed')
    assert scheduled.metadata == {'labels': ['original']}
    assert sequence.metadata == {'labels': ['original', 'sequence']}
    assert PulseSequence().metadata == {}


def test_circuit_sequence_schedule_metadata_chain():
    registry = InstSchedRegistry([0])
    template = PulseSequence(template=True)
    template.play(DriveChannel(0), Pulse(Constant(20e-9, .2)))
    registry.add(XGate(), (0,), template)
    target = Target(inst_sched_reg=registry)
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.metadata = {'nested': {'value': [1]}}
    sequence = lower(circuit, target=target)
    scheduled = sequence.schedule()
    compiled = schedule(circuit, target=target)
    assert circuit.metadata == sequence.metadata == scheduled.metadata == compiled.metadata
    sequence.metadata['nested']['value'].append(2)
    assert circuit.metadata == scheduled.metadata == compiled.metadata == {'nested': {'value': [1]}}


def test_pulse_block_preserves_sequence_metadata():
    sequence = PulseSequence(template=True, metadata={'labels': ['block']})
    sequence.play(DriveChannel(0), Pulse(Constant(20e-9, .2)))
    block = PulseBlock(sequence, 1)
    lowered = block.lower((17,))
    assert block.sequence.metadata == lowered.metadata == sequence.metadata
    lowered.metadata['labels'].append('changed')
    assert block.sequence.metadata == {'labels': ['block']}
