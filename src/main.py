from models import RandomForestModel, SignalPreProcessor
import pandas as pd
import os
import argparse
import sys


def run_once(patient_name: str, stage: str, movement: str) -> bool:
    """Run a single prediction workflow. Returns True on success, False on failure."""
    data_path = f"data/patients/{patient_name}/{stage}/{movement}.csv"
    if not os.path.exists(data_path):
        print(
            f"O arquivo especificado não existe: {data_path}. Tente novamente...")
        return False

    try:
        signal = pd.read_csv(data_path)

        preprocessor = SignalPreProcessor(signal, patient_name)
        signal_features = preprocessor.preprocess()
        preprocessed_signal = preprocessor.get_signal()

        rf_model = RandomForestModel()
        rf_model.load_model()
        rf_model.predict_and_plot(
            preprocessed_signal,
            signal_features,
            patient_name=patient_name,
            stage=stage,
            movement=movement
        )
        return True
    except Exception as e:
        print(f"Erro ao processar {data_path}: {e}")
        return False


def prompt_with_default(prompt: str, default: str) -> str:
    resp = input(f"{prompt} [{default}]: ")
    if resp.strip() == "":
        return default
    return resp.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run EMG model prediction for a patient movement")
    parser.add_argument("--patient", "-p", default="P1",
                        help="Patient folder name (default: P1)")
    parser.add_argument("--stage", "-s", default="POS1",
                        help="Stage folder name (default: POS1)")
    parser.add_argument("--movement", "-m", default="P1_alcancarbola_POS1",
                        help="Movement csv filename without extension (default: P1_alcancarbola_POS1)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (non-interactive)")
    args = parser.parse_args(argv)

    if args.once:
        success = run_once(args.patient, args.stage, args.movement)
        sys.exit(0 if success else 1)

    # Interactive loop: allow user to run multiple tests without restarting
    print("Interactive mode. Press Ctrl+C or type 'exit' to quit.")
    default_patient = args.patient
    default_stage = args.stage
    default_movement = args.movement

    try:
        while True:
            patient_name = prompt_with_default("Patient name", default_patient)
            if patient_name.lower() in ("q", "quit", "exit"):
                break
            stage = prompt_with_default("Stage", default_stage)
            if stage.lower() in ("q", "quit", "exit"):
                break
            movement = prompt_with_default(
                "Movement (filename without .csv)", default_movement)
            if movement.lower() in ("q", "quit", "exit"):
                break

            print(
                f"Running for patient={patient_name} stage={stage} movement={movement}...")
            run_once(patient_name, stage, movement)
            print(
                "Run finished. You can run another or type 'exit' when prompted to quit.\n")

    except KeyboardInterrupt:
        print("\nExiting interactive mode.")


if __name__ == "__main__":
    main()
