import warnings
warnings.filterwarnings("ignore")

import joblib
obj = joblib.load("models/general_model.pkl")
print(obj.keys())
print(obj["model_name"])