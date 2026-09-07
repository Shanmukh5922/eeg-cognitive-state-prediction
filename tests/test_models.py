import numpy as np
import pandas as pd

from src.models.general_model import prepare_data, train_general_models
from src.models.personalized_model import create_personalized_features


def make_feature_data():
	rows = []
	for subject_id in range(1, 7):
		for condition, offset in (("rest", 0.0), ("task", 1.0)):
			rows.append({
				"subject_id": subject_id,
				"session_id": 1,
				"window_id": f"{subject_id}-{condition}",
				"condition": condition,
				"alpha_power": subject_id + offset,
				"beta_power": subject_id * 0.5 + offset,
			})
	return pd.DataFrame(rows)


def test_prepare_data_excludes_metadata():
	data = make_feature_data()
	features, labels, columns = prepare_data(data)

	assert columns == ["alpha_power", "beta_power"]
	assert features.shape == (12, 2)
	assert labels.tolist().count("rest") == 6


def test_general_model_trains_on_person2_features():
	data = make_feature_data()
	model, model_name, feature_columns, X_test, y_test, results = (
		train_general_models(data)
	)

	assert model_name in results
	assert feature_columns == ["alpha_power", "beta_power"]
	assert len(model.predict(X_test)) == len(y_test)


def test_personalized_features_are_baseline_relative():
	data = make_feature_data().iloc[:2]
	baseline = pd.Series({
		"alpha_power": 1.0,
		"beta_power": 0.5,
	})

	personalized = create_personalized_features(
		data,
		baseline,
		["alpha_power", "beta_power"]
	)

	np.testing.assert_allclose(
		personalized.iloc[1].to_numpy(),
		[1.0, 1.0]
	)
