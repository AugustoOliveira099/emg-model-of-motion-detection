from RandomForestModelTrainer import RandomForestModelTrainer
from SignalPreProcessor import SignalPreProcessor
import pandas as pd
import os

patient_name = "P1"
stage = "POS1"
movement = "P1_alcancarbola_POS1"

if not os.path.exists(f"data/patients/{patient_name}/{stage}/{movement}.csv"):
    print("O arquivo especificado não existe.")
    exit(1)

# Load the data
data_path = f"data/patients/{patient_name}/{stage}/{movement}.csv"
signal = pd.read_csv(data_path)

preprocessor = SignalPreProcessor(signal)
signal_features = preprocessor.preprocess()
preprocessed_signal = preprocessor.get_signal()

rf_model = RandomForestModelTrainer()
rf_model.load_model()
rf_model.predict_and_plot(preprocessed_signal, signal_features)
