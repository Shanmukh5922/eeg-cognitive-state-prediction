"""Frequency-band power analysis for processed EEG segments."""

from __future__ import annotations

from typing import Sequence

import numpy as np


BANDS: dict[str, tuple[float, float]] = {
	"delta": (1.0, 4.0),
	"theta": (4.0, 8.0),
	"alpha": (8.0, 13.0),
	"beta": (13.0, 30.0),
	"gamma": (30.0, 40.0),
}


def _validate(data: np.ndarray, sampling_frequency: float) -> np.ndarray:
	values = np.asarray(data, dtype=float)
	if values.ndim != 3:
		raise ValueError("data must have shape (segments, channels, samples)")
	if sampling_frequency <= 0:
		raise ValueError("sampling_frequency must be positive")
	if not np.isfinite(values).all():
		raise ValueError("data must contain only finite values")
	return values


def band_powers(
	data: np.ndarray,
	sampling_frequency: float,
	*,
	relative: bool = False,
	bands: dict[str, tuple[float, float]] = BANDS,
) -> tuple[np.ndarray, tuple[str, ...]]:
	"""Calculate absolute or 1-40 Hz relative power per segment/channel/band."""

	values = _validate(data, sampling_frequency)
	frequencies = np.fft.rfftfreq(values.shape[-1], 1.0 / sampling_frequency)
	spectrum = np.abs(np.fft.rfft(values - values.mean(axis=-1, keepdims=True), axis=-1)) ** 2
	spectrum /= sampling_frequency * max(values.shape[-1], 1)
	names = tuple(bands)
	result = np.zeros((*values.shape[:2], len(names)), dtype=float)
	for band_index, (low, high) in enumerate(bands.values()):
		if low < 0 or high <= low or high > sampling_frequency / 2:
			raise ValueError("band limits must be within the Nyquist frequency")
		mask = (frequencies >= low) & (frequencies < high)
		band_spectrum = spectrum[..., mask]
		band_frequency = frequencies[mask]
		if band_spectrum.shape[-1] > 1:
			result[..., band_index] = np.sum(
				(band_spectrum[..., 1:] + band_spectrum[..., :-1])
				* np.diff(band_frequency),
				axis=-1,
			) / 2.0
	if relative:
		total = result.sum(axis=-1, keepdims=True)
		result = np.divide(result, total, out=np.zeros_like(result), where=total > 0)
	return result, names


def band_power_table(
	data: np.ndarray,
	sampling_frequency: float,
	channel_names: Sequence[str] | None = None,
	*,
	relative: bool = True,
) -> list[dict[str, float | int | str]]:
	"""Return one long-format row for each segment, channel, and band."""

	powers, names = band_powers(data, sampling_frequency, relative=relative)
	channels = list(channel_names or [f"channel_{i}" for i in range(powers.shape[1])])
	if len(channels) != powers.shape[1]:
		raise ValueError("channel_names must match the channel dimension")
	return [
		{
			"segment_index": segment_index,
			"channel_index": channel_index,
			"channel": channels[channel_index],
			"band": band,
			"power": float(powers[segment_index, channel_index, band_index]),
		}
		for segment_index in range(powers.shape[0])
		for channel_index in range(powers.shape[1])
		for band_index, band in enumerate(names)
	]
