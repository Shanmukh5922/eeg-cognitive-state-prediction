# Preprocessing data dictionary

The preprocessing pipeline saves one compressed NumPy archive (`.npz`) per recording in `data/processed/`. Raw BDF files remain local and are ignored by Git.

## Archive fields

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `data` | `(segments, channels, samples)` float array | Filtered EEG segment values in volts. |
| `metadata` | one JSON string per segment | Traceability and quality metadata described below. |

## Segment metadata

| Field | Meaning |
| --- | --- |
| `subject_id` | Dataset subject identifier, such as `sub-1`. |
| `session_id` | Dataset session identifier, such as `ses-1`. |
| `condition` | Event condition: `rs`, `easy`, `medium`, or `diff`. |
| `segment_id` | Zero-based identifier within the processed recording. |
| `start_seconds` / `end_seconds` | Segment timing relative to the recording start. |
| `sampling_frequency` | Sampling frequency in Hz, expected to be 500. |
| `channel_names` | Ordered EEG channel names matching the data axis. |
| `is_anomalous` | Whether quality checks flagged the segment. |
| `anomaly_reasons` | List of review reasons; flagged data is not silently deleted. |

Default segmentation uses complete, non-overlapping two-second windows inside each event. The default filter is 1-40 Hz with an optional 50 Hz notch. These settings are configurable and should be recorded when changed for an experiment.
