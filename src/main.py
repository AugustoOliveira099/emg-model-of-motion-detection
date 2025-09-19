from utils import analyze_patient
import argparse
import sys


def run_once(patient_name: str) -> bool:
    """Run a single prediction workflow. Returns True on success, False on failure."""
    try:
        analyze_patient(patient_name)
        return True
    except Exception as e:
        print(f"Erro ao processar {patient_name}: {e}")
        return False


def prompt_with_default(prompt: str, default: str) -> str:
    resp = input(f"{prompt} [{default}]: ")
    if resp.strip() == "":
        return default
    return resp.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run EMG model prediction for a patient movement")
    parser.add_argument("--patient", "-p", default="P4",
                        help="Patient folder name (default: P4)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (non-interactive)")
    args = parser.parse_args(argv)

    if args.once:
        success = run_once(args.patient)
        sys.exit(0 if success else 1)

    # Interactive loop: allow user to run multiple tests without restarting
    print("Interactive mode. Press Ctrl+C or type 'exit' to quit.")
    default_patient = args.patient

    try:
        while True:
            patient_name = prompt_with_default("Patient name", default_patient)
            if patient_name.lower() in ("q", "quit", "exit"):
                break
            run_once(patient_name)
            print(
                "Run finished. You can run another or type 'exit' when prompted to quit.\n")

    except KeyboardInterrupt:
        print("\nExiting interactive mode.")


if __name__ == "__main__":
    main()
