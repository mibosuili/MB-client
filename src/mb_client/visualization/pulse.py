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

"""Text and Matplotlib rendering for pulse sequences and schedules."""

from __future__ import annotations
import math
import warnings
from typing import Iterable, TypeAlias
import numpy as np
import numpy.typing as npt
from mb_client.common.parameters import Parameter, ParameterExpression
from mb_client.pulse.envelopes import Envelope
from mb_client.pulse.channels import Channel, FluxChannel, ReadoutChannel, channel_name
from mb_client.pulse.sequence import Pulse, Play, Delay, Barrier, Acquire, ShiftPhase, _finite_float, _time_array, _duration_mask
from mb_client.pulse.schedule import ScheduledPulse, ScheduledAcquire

SYMBOLIC: TypeAlias = Parameter | ParameterExpression


def _format_time_ns(value: float) -> str:
    if isinstance(value, SYMBOLIC):
        return str(value)
    return f"{value * 1e9:.3f} ns"


def _format_frequency(value: float) -> str:
    if isinstance(value, SYMBOLIC):
        return str(value)
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3g} GHz"
    return f"{value / 1e6:.3g} MHz"


def _format_ns_value(value: float) -> str:
    return f"{value * 1e9:.9g}"


def _format_time_window_ns(start: float, end: float) -> str:
    return f"[{_format_ns_value(start)}, {_format_ns_value(end)}] ns"


def _format_pulse_field(value) -> str:
    if isinstance(value, SYMBOLIC):
        return str(value)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{float(value):.12g}"
    return repr(value)


def _format_amplitude(value) -> str:
    if isinstance(value, SYMBOLIC):
        return _format_pulse_field(value)
    return f"{float(value):.3g}"


def _format_envelope(envelope: Envelope) -> str:
    details = [f"duration={_format_pulse_field(envelope.duration)}"]
    for field in ("beta", "sigma", "detuning", "omega_0"):
        if hasattr(envelope, field):
            details.append(
                f"{field}={_format_pulse_field(getattr(envelope, field))}")
    return f"{type(envelope).__name__}({', '.join(details)})"


def _format_pulse(pulse: Pulse) -> str:
    return ("Pulse("
            f"envelope={_format_envelope(pulse.envelope)}, "
            f"amplitude={_format_amplitude(pulse.envelope.amplitude)}, "
            f"frequency={_format_frequency(pulse.frequency)}, "
            f"phase={_format_pulse_field(pulse.phase)}, "
            f"offset={_format_pulse_field(pulse.offset)})")


def draw_schedule(
    self,
    *,
    t: npt.ArrayLike | None = None,
    sample_rate: float = 2e9,
    channels: Iterable[Channel] | None = None,
    group_channel: bool = True,
    with_carrier: bool = False,
    max_pulse_duration: float | None = 100e-9,
    figsize: tuple[float, float] | None = None,
):
    """Draw scheduled I/Q waveforms in compact, frameless QuTiP-QIP-style rows.

    Args:
        t: Optional absolute time points. When omitted, a time axis is
            generated using ``sample_rate``.
        sample_rate: Sampling rate used when ``t`` is omitted.
        channels: Channels to draw. The schedule's channel order is used by
            default.
        group_channel: When true, channels with the same device prefix
            (for example ``q0.xy``, ``q0.z``, and ``q0.ro``) are drawn on
            one axis. When false, one axis is created per channel.
        with_carrier: Include carrier oscillations when true. When false,
            fill the I/Q envelopes to zero using related channel colors.
        max_pulse_duration: Maximum displayed duration in seconds for a connected
            range of long pulses or acquisitions on selected channels. Only
            individual intervals exceeding this duration participate in merging
            touching or overlapping ranges. Each merged range uses one linear
            scale factor. Short pulses do not extend or bridge these ranges;
            timing within a scaled range follows the shared time axis.
            Set to None to disable scaling.
        figsize: Optional Matplotlib figure size.

    Returns:
        The Matplotlib ``(figure, axes)`` pair. ``axes`` is always a
        one-dimensional NumPy array.

    Raises:
        ValueError: If a Z-channel waveform has a nonzero Q component.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.ticker import FormatStrFormatter, NullLocator

    selected = self.channels if channels is None else tuple(channels)
    if not selected:
        raise ValueError("cannot draw a schedule without channels")
    if len(set(selected)) != len(selected):
        raise ValueError("draw channels must be unique")
    unknown = set(selected).difference(self.channels)
    if unknown:
        raise ValueError(f"unknown draw channels: {sorted(unknown, key=str)}")
    if not isinstance(group_channel, bool):
        raise TypeError("group_channel must be a bool")
    if not isinstance(with_carrier, bool):
        raise TypeError("with_carrier must be a bool")
    if max_pulse_duration is not None:
        max_pulse_duration = _finite_float(max_pulse_duration, "max_pulse_duration")
        if max_pulse_duration <= 0:
            raise ValueError("max_pulse_duration must be greater than zero")

    if t is None:
        sample_rate = _finite_float(sample_rate, "sample_rate")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        count = max(2, int(math.ceil(self.duration * sample_rate)) + 1)
        values = np.linspace(0.0, self.duration, count)
    else:
        values = _time_array(t)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("draw time values must be a non-empty 1D array")

    # Inspect overlaps in physical time, independently of visual compression.
    for channel in selected:
        pulses = sorted(
            (entry for entry in self
             if isinstance(entry, ScheduledPulse) and entry.channel == channel),
            key=lambda entry: entry.start)
        conflicts = []
        for index, entry in enumerate(pulses):
            tolerance = max(1e-15, abs(entry.pulse.duration) * 1e-12)
            for following in pulses[index + 1:]:
                if following.start > entry.end + tolerance:
                    break
                start = following.start
                end = min(entry.end, following.end)
                left, right = entry.pulse, following.pulse
                fields = []
                # Include endpoints and interior probes even for a sparse plot grid.
                probe_end = max(start, end)
                times = np.unique(np.concatenate((
                    values[(values >= start) & (values <= probe_end)],
                    np.linspace(start, probe_end, 257))))
                left_values = left.envelope.sample(times - entry.start)
                right_values = right.envelope.sample(times - following.start)
                active = (~np.isclose(left_values, 0, rtol=0, atol=1e-12)
                          & ~np.isclose(right_values, 0, rtol=0, atol=1e-12))
                if not np.any(active):
                    continue
                if not np.allclose(left_values[active], right_values[active],
                                   rtol=1e-9, atol=1e-12):
                    fields.append("envelope amplitude")
                if not np.isclose(left.frequency, right.frequency, rtol=1e-12, atol=1e-9):
                    fields.append("carrier frequency")
                if not np.isclose(left.phase, right.phase, rtol=1e-9, atol=1e-12):
                    fields.append("phase")
                if fields:
                    location = (f"share an endpoint at {start * 1e9:g} ns"
                                if abs(end - start) <= tolerance else
                                f"overlap on [{start * 1e9:g}, {end * 1e9:g}] ns")
                    conflicts.append(f"{location} with different {', '.join(fields)}")
        if conflicts:
            warnings.warn(
                f"Pulses on {channel} " + "; ".join(conflicts)
                + ". Overlapping samples are added and may produce plot artifacts.",
                UserWarning, stacklevel=2)

    scaled_intervals = []
    if max_pulse_duration is not None:
        intervals = [
            (entry.start, entry.end, max_pulse_duration)
            for entry in self
            if entry.channel in selected
            and entry.end - entry.start > max_pulse_duration
        ]
        # Only long pulses define scaling ranges. Short gates must not extend
        # a neighboring range or bridge otherwise separate long pulses.
        for start, end, displayed in sorted(intervals):
            if scaled_intervals and start <= scaled_intervals[-1][1]:
                previous = scaled_intervals[-1]
                scaled_intervals[-1] = (previous[0], max(previous[1], end),
                                    displayed)
            else:
                scaled_intervals.append((start, end, displayed))

    def display_time(time_values: np.ndarray) -> np.ndarray:
        raw = np.asarray(time_values, dtype=float)
        mapped = raw.copy()
        removed = 0.0
        for start, end, displayed in scaled_intervals:
            duration = end - start
            inside = (raw >= start) & (raw <= end)
            mapped[inside] = (start - removed +
                              (raw[inside] - start) / duration * displayed)
            mapped[raw > end] -= duration - displayed
            removed += duration - displayed
        return mapped

    if group_channel:
        plot_groups: dict[object, list[Channel]] = {}
        for channel in selected:
            device = channel.qubit_index
            plot_groups.setdefault(device, []).append(channel)
    else:
        plot_groups = {channel: [channel] for channel in selected}

    def _pulse_support_mask(channel: Channel) -> np.ndarray:
        mask = np.zeros(values.shape, dtype=bool)
        for entry in self:
            if isinstance(entry, ScheduledPulse) and entry.channel == channel:
                mask |= _duration_mask(values - entry.start,
                                       entry.pulse.duration)
        return mask

    def _validate_zero_q_component(channel: Channel, waveform: np.ndarray,
                                   support_mask: np.ndarray) -> None:
        active_imag = waveform.imag[support_mask]
        if active_imag.size and not np.allclose(
                active_imag, 0.0, rtol=1e-9, atol=1e-12):
            raise ValueError(f"channel {channel!r} has a nonzero Q component")

    if figsize is None:
        figsize = (10.0, 1.6 * len(plot_groups))
    figure, axes = plt.subplots(len(plot_groups),
                                1,
                                sharex=True,
                                sharey=True,
                                squeeze=False,
                                figsize=figsize)
    # Reserve a fixed-height band for a suptitle. Using a physical height
    # instead of a fixed fraction avoids wasting space on multi-axis plots.
    plot_top = 0.98 - 0.18 / figure.get_figheight()
    figure.subplots_adjust(left=0.1,
                           right=0.98,
                           bottom=0.45 / figure.get_figheight(),
                           top=plot_top,
                           hspace=0.12)
    axes = axes[:, 0]
    plot_time = display_time(values) * 1e9
    plot_min = float(np.nanmin(plot_time))
    plot_max = float(np.nanmax(plot_time))
    peak = 0.0
    for axis_index, (axis, (group_name, group_channels)) in enumerate(
            zip(axes, plot_groups.items())):
        for channel in group_channels:
            support_mask = _pulse_support_mask(channel)
            waveform = self.sample(channel,
                                   values,
                                   with_carrier=with_carrier)
            waveform = np.where(support_mask, waveform, np.nan + 1j * np.nan)
            components = [waveform.real]
            if not isinstance(channel, (FluxChannel, ReadoutChannel)):
                components.append(waveform.imag)
            for component in components:
                finite = component[np.isfinite(component)]
                if finite.size:
                    peak = max(peak, float(np.max(np.abs(finite))))
            if isinstance(channel, FluxChannel):
                color = "tab:red"
            elif isinstance(channel, ReadoutChannel):
                color = "tab:green"
            else:
                color = axis._get_lines.get_next_color()
            if with_carrier or isinstance(channel, FluxChannel):
                axis.plot(plot_time, waveform.real, color=color)
            if not with_carrier and not isinstance(channel, FluxChannel):
                axis.fill_between(plot_time, 0, waveform.real,
                                  color=color, alpha=0.5, linewidth=0)

            if isinstance(channel, ReadoutChannel):
                # Readout carriers can have a nonzero Q component, but
                # measurement-device channels intentionally draw I only.
                pass
            elif isinstance(channel, (FluxChannel, ReadoutChannel)):
                _validate_zero_q_component(channel, waveform, support_mask)
            else:
                q_color = tuple(0.8 * component + 0.2
                                for component in to_rgb(color))
                if with_carrier:
                    axis.plot(plot_time,
                              waveform.imag,
                              color=q_color,
                              linestyle="--")
                if not with_carrier:
                    axis.fill_between(plot_time, 0, waveform.imag,
                                      color=q_color, alpha=0.25, linewidth=0)
        label = (group_name if group_channel else group_channels[0])
        axis.set_ylabel(f"q{label}" if group_channel else channel_name(label),
                        rotation=0, ha="right", va="center", labelpad=12)
        axis.yaxis.set_minor_locator(NullLocator())
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.spines["left"].set_visible(True)
        axis.tick_params(axis="y", left=True, right=False, labelleft=True)
        axis.axhline(0.0, color="0.45", linewidth=0.7, zorder=0)
        axis.tick_params(axis="x", bottom=axis_index == len(axes) - 1)
        if plot_max > plot_min:
            axis.set_xlim(plot_min, plot_max)
        axis.grid(False)
        removed = 0.0
        tick_positions = []
        tick_labels = []
        for start, end, displayed in scaled_intervals:
            display_start = start - removed
            tick_positions.extend([display_start * 1e9,
                                   (display_start + displayed) * 1e9])
            tick_labels.extend([f"{start * 1e9:g}", f"{end * 1e9:g}"])
            removed += end - start - displayed
        if tick_positions:
            axis.set_xticks(tick_positions, labels=tick_labels)
            if plot_max > plot_min:
                axis.set_xlim(plot_min, plot_max)
    # Shared limits preserve amplitude comparisons across channels and qubits.
    axes[0].set_ylim(-1.1 * (peak or 1.0), 1.1 * (peak or 1.0))
    axes[0].set_yticks([0.0, peak or 1.0])
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    axes[-1].set_xlabel("Time (ns)")
    return figure, axes


def format_sequence(self) -> str:
    if not self._operations:
        return "PulseSequence[0]" + (" template=True (operand-relative)"
                                     if self.template else "")
    lines = [
        f"PulseSequence[{len(self._operations)}]" +
        (" template=True (operand-relative)" if self.template else "")
    ]
    for index, operation in enumerate(self._operations):
        if isinstance(operation, Play):
            line = (f"  [{index:02d}] Play    ch={operation.channel} "
                    f"pulse={_format_pulse(operation.pulse)}")
        elif isinstance(operation, Delay):
            line = (f"  [{index:02d}] Delay   ch={operation.channel} "
                    f"dur={_format_time_ns(operation.duration)}")
        elif isinstance(operation, Acquire):
            line = (f"  [{index:02d}] Acquire ch={operation.channel} "
                    f"dur={_format_time_ns(operation.duration)} "
                    f"demod={_format_frequency(operation.demod_freq)}")
        elif isinstance(operation, ShiftPhase):
            line = f"  [{index:02d}] ShiftPhase ch={operation.channel} phase={operation.phase}"
        else:
            channels = ", ".join(map(
                str, operation.channels)) if operation.channels else "*"
            line = f"  [{index:02d}] Barrier ch={channels}"
        lines.append(line)
    return "\n".join(lines)


def format_schedule(self) -> str:
    if not self._entries:
        return "PulseSchedule[0] duration=0.000 ns"
    lines = [
        f"PulseSchedule[{len(self._entries)}] duration={_format_time_ns(self.duration)}"
    ]
    for index, entry in enumerate(self._entries):
        if isinstance(entry, ScheduledPulse):
            lines.append(
                f"  [{index:02d}] t={_format_time_window_ns(entry.start, entry.end)} Play"
                f"    ch={entry.channel} pulse={_format_pulse(entry.pulse)}")
        else:
            operation = entry.operation
            lines.append(
                f"  [{index:02d}] t={_format_time_window_ns(entry.start, entry.end)} Acquire"
                f" ch={entry.channel}"
                f" demod={_format_frequency(operation.demod_freq)}")
    return "\n".join(lines)
