import os
import sys
import pandas as pd
from src.models import RandomForestModel, SignalPreProcessor

STAGES = ["PRE", "POS1", "POS2"]


def analyze_patient(patient_name):
    base_dir = os.path.join("data", "patients", patient_name)
    model = RandomForestModel()
    model.load_model()

    for stage in STAGES:
        stage_dir = os.path.join(base_dir, stage)
        if not os.path.isdir(stage_dir):
            print(f"Pasta não encontrada: {stage_dir}")
            continue
        for movement in os.listdir(stage_dir):
            if not movement.endswith(".csv"):
                continue  # Ignora .DS_Store e outros
            file_path = os.path.join(stage_dir, movement)
            print(f"Processando {file_path}")
            # Carregue o sinal EMG
            signal = pd.read_csv(file_path)
            preprocessor = SignalPreProcessor(signal, patient_name)
            signal_features = preprocessor.preprocess()
            preprocessed_signal = preprocessor.get_signal()
            model.predict_and_plot(
                preprocessed_signal,
                signal_features,
                patient_name,
                stage,
                movement
            )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze_patient.py <NOME_PACIENTE>")
        sys.exit(1)
    analyze_patient(sys.argv[1])
