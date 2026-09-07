import json

import numpy as np

from src.eda.channel_analysis import channel_difference, channel_summary
from src.eda.correlation_condition_analysis import (
	channel_correlation,
	condition_summary,
	resting_vs_cognitive,
)
from src.eda.frequency_domain import BANDS, band_powers
from src.eda.time_domain import load_processed_archive, summarize_time_domain, time_domain_features
from src.feature_selection.select_features import select_features


def make_data() -> tuple[np.ndarray, list[dict[str, object]]]:
	time = np.arange(1000) / 100.0
	data = np.zeros((4, 2, len(time)))
	data[:, 0] = np.sin(2 * np.pi * 10 * time)
	data[:, 1] = 0.5 * np.sin(2 * np.pi * 20 * time)
	data[2:] += 0.1
	metadata = [
		{"condition": "rs", "is_anomalous": False},
		{"condition": "rs", "is_anomalous": False},
		{"condition": "easy", "is_anomalous": False},
		{"condition": "easy", "is_anomalous": False},
	]
	return data, metadata


def test_time_domain_and_channel_analysis() -> None:
	data, _ = make_data()
	summary = summarize_time_domain(data, ["C3", "C4"])
	assert summary["std"].shape == (4, 2)
	assert summary["rms"][0, 0] > summary["rms"][0, 1]
	assert len(time_domain_features(data, ["C3", "C4"])) == 8
	assert len(channel_summary(data, ["C3", "C4"])) == 2
	difference = channel_difference(data, ["C3", "C4"], "C3", "C4")
	assert np.asarray(difference["difference"]).shape == (4, 1000)


def test_frequency_and_condition_analysis() -> None:
	data, metadata = make_data()
	powers, bands = band_powers(data, 100.0, relative=True)
	assert bands == tuple(BANDS)
	assert powers.shape == (4, 2, 5)
	assert np.allclose(powers.sum(axis=-1), 1.0)
	correlation = channel_correlation(data, ["C3", "C4"])
	assert correlation.shape == (2, 2)
	assert len(condition_summary(data, metadata)) == 2
	assert set(resting_vs_cognitive(data, metadata)) == {"resting", "cognitive"}


def test_archive_loader_excludes_anomalous_segments(tmp_path) -> None:
	data, metadata = make_data()
	metadata[-1]["is_anomalous"] = True
	archive = tmp_path / "segments.npz"
	np.savez_compressed(archive, data=data, metadata=np.asarray([json.dumps(item) for item in metadata]))
	clean_data, clean_metadata = load_processed_archive(archive)
	assert clean_data.shape[0] == 3
	assert len(clean_metadata) == 3


def test_feature_selection_returns_compact_final_table() -> None:
	data, metadata = make_data()
	result = select_features(data, metadata, 100.0, ["C3", "C4"], max_features=4)
	assert result.values.shape == (4, 4)
	assert len(result.feature_names) == 4
	assert set(result.scores) == set(result.feature_names)
