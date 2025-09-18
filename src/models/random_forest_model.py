from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import joblib


class RandomForestModel:
    def __init__(self):
        self.model = None

    def split_data(self, df, test_size=0.2):
        X = df.drop('InMovement', axis=1)
        y = df['InMovement']
        return train_test_split(X, y, test_size=test_size, random_state=1)

    def train(self, X_train, y_train):
        # self.model = SVC(kernel="rbf", C=1, gamma="auto")
        # param_grid = {
        #     "n_estimators": [100, 200, 300],
        #     "max_depth": [None, 10, 20],
        #     "min_samples_split": [2, 5, 10],
        #     "min_samples_leaf": [1, 2, 5],
        #     "max_features": ["sqrt", "log2"],
        # }
        self.model = RandomForestClassifier(
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=5,
            min_samples_split=2,
            n_estimators=100,
            random_state=3
        )
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def load_model(self, path='./rf_model.pkl'):
        self.model = joblib.load(path)

    def get_model(self):
        return self.model

    def save_model(self, path='./data/rf_model.pkl'):
        joblib.dump(self.model, path)

    def get_confusion_matrix(self, X_test, y_test, save_path="./assets/confusion_matrix.png"):
        y_pred = self.predict(X_test)
        # Gerar e plotar matriz
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        disp.ax_.set_title("Confusion Matrix")

        # Salvar
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def load_features_to_dataframe(self, base_path='data/extracted_features'):
        all_features = []
        for patient in os.listdir(base_path):
            patient_path = os.path.join(base_path, patient)
            if os.path.isdir(patient_path):
                for stage in os.listdir(patient_path):
                    stage_path = os.path.join(patient_path, stage)
                    if os.path.isdir(stage_path):
                        for movement_file in os.listdir(stage_path):
                            movement_path = os.path.join(
                                stage_path, movement_file)
                            try:
                                with open(movement_path, 'r') as f:
                                    movement_features = json.load(f)
                                    for muscle, windows in movement_features.items():
                                        for window_features in windows:
                                            all_features.append(
                                                window_features)
                            except json.JSONDecodeError as e:
                                print(
                                    f"Error decoding JSON in file {movement_path}: {e}")
        return pd.DataFrame(all_features)

    def predict_and_plot(self,
                         signal,
                         signal_features,
                         patient_name=None,
                         stage=None,
                         movement=None):
        predictions = {}
        times = {}
        min = 0
        max = 0
        # Determine output directory
        output_dir = 'assets'
        if patient_name and stage and movement:
            output_dir = os.path.join('assets', str(patient_name), str(stage), str(movement))
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        for muscle in signal_features:
            times[muscle] = signal_features[muscle].pop('Time')
            predictions[muscle] = self.predict(signal_features[muscle])
            newMin = signal[muscle].min()
            newMax = signal[muscle].max()
            if newMin < min:
                min = newMin
            if newMax > max:
                # Salva com uma margem de 20% para melhor visualização
                max = newMax + newMax * 20/100

        for column_time, muscle in zip(signal.columns[::2], signal.columns[1::2]):
            # for muscle, preds in predictions.items():
            plt.figure(figsize=(10, 6))
            plt.plot(signal[column_time], signal[muscle])
            preds = predictions[muscle]
            for i in range(1, len(preds)):
                if preds[i] != preds[i - 1]:
                    plt.axvline(x=times[muscle][i], color='r', linestyle='--')
                    with open(os.path.join(output_dir, f'{muscle}_movement_times.txt'), 'a') as f:
                        f.write(f'{times[muscle][i]}\n')
            # # Fixa o eixo y com o menor eo maior valor encontrado entre os músculos
            # # Alguns músculo possuem um pico de sinal, o que pode prejudicar a visualização
            # plt.ylim(min, max)
            plt.xlabel('Time')
            plt.ylabel('Signal')
            plt.title(f'Signal with Movement Predictions for {muscle}')
            plt.savefig(os.path.join(
                output_dir, f'{muscle}_movement_predictions.png'))
            plt.close()
