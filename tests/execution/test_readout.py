import numpy as np
import pytest

from mb_client.execution.readout import classify_iq, nearest_center_discriminator


def test_nearest_centers_and_little_endian_counts():
    discriminator = nearest_center_discriminator([[0j, 1j], [2+0j, 3+0j]])
    iq = np.array([[.9j, 2.1], [.1j, 2.9], [.8j, 2.2]])
    assert classify_iq(iq, discriminator) == {'01': 2, '10': 1}
    assert discriminator(np.array([.5j, 2.5])) == '00'


def test_custom_discriminator_receives_whole_shots_without_mutating_input():
    iq = np.array([[1j, 2j], [3j, 4j]])
    seen = []

    def discriminator(shot):
        seen.append(shot.copy())
        shot[:] = 0
        return '10'

    assert classify_iq(iq, discriminator) == {'10': 2}
    np.testing.assert_array_equal(seen, iq)
    assert classify_iq(np.empty((0, 2)), discriminator) == {}


def test_centers_are_copied_and_single_qubit_works():
    centers = np.array([[0j, 1j]])
    discriminator = nearest_center_discriminator(centers)
    centers[:] = 100j
    assert classify_iq([[.1j], [.9j]], discriminator) == {'0': 1, '1': 1}


@pytest.mark.parametrize('iq', [[0, 1], [[[0]]], [[np.nan]], [[np.inf]], np.empty((2, 0))])
def test_invalid_iq(iq):
    with pytest.raises(ValueError):
        classify_iq(iq, lambda shot: '0')


@pytest.mark.parametrize('centers', [[0, 1], [[0]], [[0, 1, 2]], [[0, np.nan]], []])
def test_invalid_centers(centers):
    with pytest.raises(ValueError):
        nearest_center_discriminator(centers)


def test_discriminator_contract():
    with pytest.raises(TypeError, match='callable'):
        classify_iq([[0]], None)
    for state in (0, '', '00'):
        with pytest.raises(ValueError, match='one state character'):
            classify_iq([[0]], lambda shot: state)
    with pytest.raises(ValueError, match='one finite value'):
        nearest_center_discriminator([[0, 1]])([0, 1])
