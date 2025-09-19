import os
import pandas as pd
from models import RandomForestModel, SignalPreProcessor

STAGES = ["PRE", "POS1", "POS2"]


def analyze_patient(patient_name):
    base_dir = os.path.join("data", "patients", patient_name)
    model = RandomForestModel()
    model.load_model()
    print(f"Running for patient={patient_name}")

    # Dicionário para armazenar todos os tempos
    summary_intervals = {}

    for stage in STAGES:
        print(f"Running for stage={stage}")
        stage_dir = os.path.join(base_dir, stage)
        if not os.path.isdir(stage_dir):
            print(f"Pasta não encontrada: {stage_dir}")
            continue
        for movement in os.listdir(stage_dir):
            print(f"Running for movement={movement}")
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
            # Lê o movement_times.json gerado
            output_dir = os.path.join(
                "assets", str(patient_name), str(stage), str(movement)
            )
            movement_times_path = os.path.join(output_dir, "movement_times.json")
            if os.path.exists(movement_times_path):
                import json
                with open(movement_times_path, "r") as f:
                    times_dict = json.load(f)
                # Remove extensão e estágio do nome do movimento para chave
                movement_key = movement.replace(f"_{stage}.csv", "")
                if movement_key not in summary_intervals:
                    summary_intervals[movement_key] = {}
                for idx, times in times_dict.items():
                    if idx not in summary_intervals[movement_key]:
                        summary_intervals[movement_key][idx] = {}
                    summary_intervals[movement_key][idx][stage] = times

    # Salva o resumo final após todos os movimentos
    summary_path = os.path.join("assets", patient_name, f"{patient_name}_summary_movement_intervals.json")
    with open(summary_path, "w") as f:
        import json
        json.dump(summary_intervals, f, indent=2)
    print(f"Resumo salvo em {summary_path}")
