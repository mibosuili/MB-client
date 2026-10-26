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

"""Pulse construction and scheduling primitives."""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
from typing import TypeAlias
import numpy as np
import numpy.typing as npt
from scipy.special import erf
from mb_client.common.parameters import Parameter, ParameterExpression

SYMBOLIC: TypeAlias = Parameter | ParameterExpression


def _template_value(value, name: str):
    if isinstance(value, SYMBOLIC):
        return value
    return _finite_float(value, name)


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _time_array(t: npt.ArrayLike) -> np.ndarray:
    try:
        result = np.asarray(t, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("time values must be real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("time values must be finite")
    return result


def _duration_mask(t: np.ndarray, duration: float) -> np.ndarray:
    tolerance = max(1e-15, abs(duration) * 1e-12)
    return (t >= -tolerance) & (t <= duration + tolerance)


class Envelope(ABC):
    r"""Immutable normalized complex baseband envelope.

    Every envelope represents

    .. math::

        E(t) = I(t) + iQ(t),

    where :math:`I(t)` and :math:`Q(t)` are the in-phase and quadrature
    components. An envelope with duration :math:`T` has support on
    :math:`0 \leq t \leq T` and is zero outside that interval.

    Subclasses should assign any additional attributes before calling
    ``super().__init__``. The base initializer freezes the instance.
    """

    def __init__(self, duration: float, amplitude: float):
        duration = _template_value(duration, "duration")
        amplitude = _template_value(amplitude, "amplitude")
        if not isinstance(duration, SYMBOLIC) and duration <= 0:
            raise ValueError("duration must be greater than zero")
        if not isinstance(amplitude, SYMBOLIC) and amplitude < 0:
            raise ValueError("amplitude must be positive")
        object.__setattr__(self, "_duration", duration)
        object.__setattr__(self, "_amplitude", amplitude)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{type(self).__name__} is immutable")
        object.__setattr__(self, name, value)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @abstractmethod
    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        """Evaluate at local times measured from the beginning of the pulse."""

    def _checked_samples(self, t: np.ndarray, samples) -> np.ndarray:
        result = np.asarray(samples, dtype=complex)
        if result.shape != t.shape:
            raise ValueError(
                f"{type(self).__name__}.sample() returned shape {result.shape}; "
                f"expected {t.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(
                f"{type(self).__name__}.sample() returned non-finite values")
        return result


class Constant(Envelope):
    r"""Unit constant envelope.

    For duration :math:`T`, the complex envelope is

    .. math::

        E(t) = I(t) + iQ(t) =
        \begin{cases}
            1, & 0 \leq t \leq T, \\
            0, & \text{otherwise}.
        \end{cases}

    Thus :math:`I(t)=1` and :math:`Q(t)=0` over the pulse support.
    """

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        result = _duration_mask(values, self.duration).astype(complex)
        result *= self.amplitude
        return self._checked_samples(values, result)


class Idle(Envelope):
    r"""Idle envelope with zero amplitude.

    For duration :math:`T`, the complex envelope is

    .. math::

        E(t) = I(t) + iQ(t) =
        \begin{cases}
            0, & 0 \leq t \leq T, \\
            0, & \text{otherwise}.
        \end{cases}

    Thus :math:`I(t)=0` and :math:`Q(t)=0` over the pulse support.
    """

    def __init__(self, duration: float):
        super().__init__(duration, amplitude=0.0)

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        mask = _duration_mask(values, self.duration)
        result = np.zeros_like(values, dtype=complex)
        return self._checked_samples(values, result * mask)


class Cos(Envelope):
    r"""Raised-cosine envelope with zero quadrature component.

    For duration :math:`T`, define

    .. math::

        I(t) = \frac{1}{2}\left[1 +
        \cos\left(\frac{2\pi}{T}\left(t-\frac{T}{2}\right)\right)\right],
        \qquad Q(t)=0,

    for :math:`0 \leq t \leq T`. Both components are zero outside that
    interval, and :math:`E(t)=I(t)+iQ(t)`.
    """

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        mask = _duration_mask(values, self.duration)
        result = 0.5 * (1 + np.cos(2 * np.pi / self.duration *
                                   (values - self.duration / 2)))
        result *= self.amplitude
        return self._checked_samples(values, result * mask)


class Gaussian(Envelope):
    r"""Endpoint-subtracted, unit-height Gaussian envelope.

    The implementation follows the legacy cutoff convention :math:`c=2`.
    For duration :math:`T`, let :math:`\sigma=T/(2c)=T/4` and
    :math:`\mu=T/2`. Then

    .. math::

        I(t) = \frac{
            \exp\left[-\frac{(t-\mu)^2}{2\sigma^2}\right]
            - \exp\left[-\frac{c^2}{2}\right]
        }{1 - \exp\left[-\frac{c^2}{2}\right]},
        \qquad Q(t)=0,

    for :math:`0 \leq t \leq T`. Both components are zero outside that
    interval. The subtraction makes the endpoints zero and the denominator
    normalizes the center to one.
    """

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        cutoff = 2.0
        sigma = self.duration / (2 * cutoff)
        center = cutoff * sigma
        offset = np.exp(-(cutoff**2) / 2)
        result = (np.exp(-((values - center)**2) /
                         (2 * sigma**2)) - offset) / (1 - offset)
        result *= self.amplitude
        mask = _duration_mask(values, self.duration)
        return self._checked_samples(values, result * mask)


class NetZero(Envelope):
    r"""Bipolar error-function envelope with approximately zero net area.

    ``sigma`` controls the transition width between the positive and negative
    halves.  The definition matches the legacy ``pulse_shape.net_zero``
    waveform over the pulse support.
    """

    def __init__(self,
                 duration: float,
                 amplitude: float,
                 sigma: float = 0.1e-9):
        sigma = _template_value(sigma, "sigma")
        if not isinstance(sigma, SYMBOLIC) and sigma <= 0:
            raise ValueError("sigma must be greater than zero")
        object.__setattr__(self, "_sigma", sigma)
        super().__init__(duration, amplitude)

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        sigma = self.sigma * np.sqrt(2)
        result = 0.5 * (erf((values / sigma) - 2 * np.sqrt(2)) - 2 * erf(
            (values - self.duration / 2) / sigma) + erf((
                (values - self.duration) / sigma) + 2 * np.sqrt(2)))
        result *= self.amplitude
        mask = _duration_mask(values, self.duration)
        return self._checked_samples(values, result * mask)


class Drag(Envelope):
    r"""Raised-cosine envelope with a derivative quadrature correction.

    For duration :math:`T`, define

    .. math::

        \theta(t) = \frac{2\pi}{T}\left(t-\frac{T}{2}\right),

        I(t) = \frac{1}{2}\left[1+\cos\theta(t)\right],
        \qquad
        Q(t) = \beta\,\frac{\pi}{T\,10^9}\sin\theta(t),

    for :math:`0 \leq t \leq T`, with :math:`E(t)=I(t)+iQ(t)`. Both
    components are zero outside that interval. The :math:`10^9` factor retains
    the legacy convention in which ``beta`` (``drag_scaling``) scales the
    derivative with time expressed in nanoseconds.
    """

    def __init__(self,
                 duration: float,
                 amplitude: float,
                 beta: float = 0.0,
                 detuning: float = 0.0):
        object.__setattr__(self, "_detuning",
                           _template_value(detuning, "detuning"))
        object.__setattr__(self, "_beta", _template_value(beta, "beta"))
        super().__init__(duration, amplitude)

    @property
    def detuning(self) -> float:
        return self._detuning

    @property
    def beta(self) -> float:
        return self._beta

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        angle = 2 * np.pi / self.duration * (values - self.duration / 2)
        i_quad = 0.5 * (1 + np.cos(angle))
        q_quad = (self.beta * 0.5 * np.sin(angle) * 2 * np.pi /
                  (self.duration * 1e9))
        result = (i_quad + 1j * q_quad) * np.exp(
            -2j * np.pi * self.detuning * values)
        result *= self.amplitude
        mask = _duration_mask(values, self.duration)
        return self._checked_samples(values, result * mask)


class ErrorFunction(Envelope):
    """Smoothed rectangular readout envelope."""

    def __init__(
        self,
        duration: float,
        amplitude: float,
        sigma: float = 2e-9,
        ramp_up: float = 0,
        ramp_down: float = 0,
        up_length: float = 100e-9,
        down_length: float = 50e-9,
    ):
        sigma = _template_value(sigma, "sigma")
        if not isinstance(sigma, SYMBOLIC) and sigma <= 0:
            raise ValueError("sigma must be greater than zero")
        object.__setattr__(self, "_sigma", sigma)
        object.__setattr__(self, "_ramp_up",
                           _template_value(ramp_up, "ramp_up"))
        object.__setattr__(self, "_ramp_down",
                           _template_value(ramp_down, "ramp_down"))
        object.__setattr__(self, "_up_length",
                           _template_value(up_length, "up_length"))
        object.__setattr__(self, "_down_length",
                           _template_value(down_length, "down_length"))
        super().__init__(duration, amplitude)

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def ramp_up(self) -> float:
        return self._ramp_up

    @property
    def ramp_down(self) -> float:
        return self._ramp_down

    @property
    def up_length(self) -> float:
        return self._up_length

    @property
    def down_length(self) -> float:
        return self._down_length

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        shifted = values - 4 * self.sigma
        plateau = self.duration - 16 * self.sigma - self.down_length
        width = np.sqrt(2) * self.sigma

        result = self.amplitude / 2 * (erf(shifted / width) - erf(
            (shifted - plateau) / width))

        result += self.ramp_up * self.amplitude / 2 * (
            erf(shifted / width) - erf((shifted - self.up_length) / width))

        result -= self.ramp_down * self.amplitude / 2 * (erf(
            (shifted - plateau - 8 * self.sigma) / width) - erf(
                (shifted - plateau - self.down_length - 8 * self.sigma) /
                width))

        mask = _duration_mask(values, self.duration)
        return self._checked_samples(values, result * mask)


class Tanh(Envelope):
    """Smoothed flux envelope."""

    def __init__(self,
                 duration: float,
                 amplitude: float,
                 sigma: float = 0.1e-9,
                 cutoff: float = 2.0):
        sigma = _template_value(sigma, "sigma")
        cutoff = _template_value(cutoff, "cutoff")
        if not isinstance(sigma, SYMBOLIC) and sigma <= 0:
            raise ValueError("sigma must be greater than zero")
        object.__setattr__(self, "_sigma", sigma)
        object.__setattr__(self, "_cutoff", cutoff)
        super().__init__(duration, amplitude)

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        sigma = min(self.sigma, self.duration / (2.5 * self.cutoff))

        rise_center = self.cutoff * sigma
        fall_center = self.duration - self.cutoff * sigma
        offset = 0.5 * (np.tanh(-self.cutoff) +
                        np.tanh(self.duration / sigma - self.cutoff))
        peak = np.tanh(self.duration / (2 * sigma) - self.cutoff)

        result = self.amplitude * (0.5 * np.tanh(
            (values - rise_center) / sigma) + 0.5 * np.tanh(
                (fall_center - values) / sigma) - offset) / (peak - offset)

        mask = _duration_mask(values, self.duration)
        return self._checked_samples(values, result * mask)


class ModulatedSin(Envelope):
    """Phase-modulated sine envelope for spectroscopy."""

    def __init__(
        self,
        duration: float,
        amplitude: float,
        omega_0: float = 50e6,
    ):
        object.__setattr__(self, "_omega_0",
                           _template_value(omega_0, "omega_0"))
        super().__init__(duration, amplitude)

    @property
    def omega_0(self) -> float:
        return self._omega_0

    def sample(self, t: npt.ArrayLike) -> np.ndarray:
        values = _time_array(t)
        sine = np.sin(np.pi * values / self.duration)

        result = self.amplitude * sine * np.exp(
            -2j * self.omega_0 * self.duration * sine)
        mask = _duration_mask(values, self.duration)
        return self._checked_samples(values, result * mask)


class UnsupportedEnvelopeError(TypeError):
    """Raised when an envelope cannot be recognized."""


__all__ = [
    "Constant",
    "Cos",
    "Idle",
    "Drag",
    "Envelope",
    "Gaussian",
    "NetZero",
    "ErrorFunction",
    "Tanh",
    "ModulatedSin",
    "UnsupportedEnvelopeError",
]
