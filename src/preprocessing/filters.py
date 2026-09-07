"""Configurable filters for continuous EEG recordings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FilterConfig:
	"""Default frequency settings for the Neuroergonomic EEG recordings."""

	low_cutoff: float = 1.0
	high_cutoff: float = 40.0
	notch_frequency: float | None = 50.0
	notch_width: float = 2.0


def _validate_config(config: FilterConfig, sampling_frequency: float) -> None:
	"""Validate filter settings before MNE mutates a recording."""

	nyquist = sampling_frequency / 2.0
	if config.low_cutoff < 0:
		raise ValueError("low_cutoff must be non-negative")
	if config.high_cutoff <= config.low_cutoff:
		raise ValueError("high_cutoff must be greater than low_cutoff")
	if config.high_cutoff >= nyquist:
		raise ValueError(
			f"high_cutoff must be below the Nyquist frequency ({nyquist:g} Hz)"
		)
	if config.notch_width <= 0:
		raise ValueError("notch_width must be positive")
	if config.notch_frequency is not None:
		if config.notch_frequency <= 0:
			raise ValueError("notch_frequency must be positive")
		if config.notch_frequency >= nyquist:
			raise ValueError(
				f"notch_frequency must be below the Nyquist frequency ({nyquist:g} Hz)"
			)


def filter_raw(
	raw: Any,
	*,
	config: FilterConfig | None = None,
	copy: bool = True,
	picks: Any = "eeg",
	verbose: str | bool | None = None,
) -> Any:
	"""Apply EEG band-pass and optional power-line filtering.

	The default 1-40 Hz band-pass retains common EEG rhythms while reducing
	slow drift and high-frequency noise.  The optional 50 Hz notch targets the
	dataset's stated power-line frequency.  MNE's zero-phase FIR filters are
	used through ``Raw.filter`` and ``Raw.notch_filter``.

	Parameters
	----------
	raw:
		An MNE ``Raw`` instance.
	config:
		Frequency settings. Defaults to :class:`FilterConfig`.
	copy:
		Return a copy when true (the default); when false, modify ``raw``.
	picks:
		MNE channel selection passed to both filtering operations.
	"""

	if not hasattr(raw, "info") or not hasattr(raw, "copy"):
		raise TypeError("raw must be an MNE Raw-like object")
	filter_config = config or FilterConfig()
	sampling_frequency = float(raw.info["sfreq"])
	_validate_config(filter_config, sampling_frequency)
	filtered = raw.copy() if copy else raw
	if not filtered.preload:
		filtered.load_data()

	filtered.filter(
		l_freq=filter_config.low_cutoff,
		h_freq=filter_config.high_cutoff,
		picks=picks,
		verbose=verbose,
	)
	if filter_config.notch_frequency is not None:
		filtered.notch_filter(
			freqs=[filter_config.notch_frequency],
			notch_widths=filter_config.notch_width,
			picks=picks,
			verbose=verbose,
		)
	return filtered
