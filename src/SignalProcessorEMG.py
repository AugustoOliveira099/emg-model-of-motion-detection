import os
import json
import numpy as np
from scipy.signal import iirnotch, filtfilt, butter

class SignalProcessorEMG:
  def __init__(self, movement_data):
    """
    Initialize a SignalProcessorEMG object.

    Args:
      movement_data (dict): A dictionary containing movement data for each stage.
    """
    self.features = {}
    self.window_time_ms = 100
    self.movement_data = movement_data
    self.data_columns, self.time_columns = self.__get_column_names()

  def get_movement_signals(self):
    return self.movement_data
  
  def apply_notch_filter(self, fs = 1926, f0 = 60, Q = 70):
    print("Applying notch filter")
    for stage in self.movement_data.keys():
      for movement in self.movement_data[stage].keys():
        for column in self.data_columns:
          self.movement_data[stage][movement][column] = self.__notch_filter(self.movement_data[stage][movement][column], fs = fs, f0 = f0, Q = Q)

  def apply_bandpass_filter(self, fs = 1926, lowcut = 20, highcut = 450, order = 4):
    print("Applying bandpass filter")
    for stage in self.movement_data.keys():
      for movement in self.movement_data[stage].keys():
        for column in self.data_columns:
          self.movement_data[stage][movement][column] = self.__bandpass_filter(self.movement_data[stage][movement][column], N = order, freq_low = lowcut, freq_high = highcut, fs = fs)

  def apply_full_wave_rectification(self):
    print("Applying full wave rectification")
    for stage in self.movement_data.keys():
      for movement in self.movement_data[stage].keys():
        for column in self.data_columns:
          self.movement_data[stage][movement][column] = np.abs(self.movement_data[stage][movement][column])

  def capture_signal_segment(self, initial_time=5, final_time=25):
    print("Capturing signal segment")
    for stage in self.movement_data.keys():
      for movement in self.movement_data[stage].keys():
        for column in self.time_columns:
          self.movement_data[stage][movement] = self.__split_df(
            self.movement_data[stage][movement], column, initial_time, final_time
          )

  def windowing_data(self, window_time_ms = 100, fs = 1926):
    print("Windowing data")
    self.window_time_ms = window_time_ms
    for stage in self.movement_data.keys():
      for movement in self.movement_data[stage].keys():
        windowed_dict = {}
        for column in self.data_columns:
          windowed_data = self.__windowing_data(self.movement_data[stage][movement][column], window_time_ms, fs)
          windowed_dict[column] = windowed_data
        self.movement_data[stage][movement] = windowed_dict

  def extract_features(self, movement_intervals):
    print("Extracting features")
    for stage in self.movement_data.keys():
      print(stage)
      self.features[stage] = {}
      for _movement in movement_intervals.keys():
        movement = f"{_movement}{stage}"
        self.features[stage][movement] = {}
        for column_index in movement_intervals[_movement].keys():
          column = self.data_columns[column_index]
          print(self.movement_data[stage].keys())
          windows = self.movement_data[stage][movement][column]
          self.features[stage][movement][column] = [self.__extract_window_features(window) for window in windows]
          self.__add_movement_feature(windows, movement_intervals, stage, _movement, column, column_index)

  def __add_movement_feature(self, windows, movement_intervals, stage, _movement, column, column_index):
    intervals = movement_intervals.get(_movement, {}).get(column_index, {}).get(stage, [])
    movement_feature = []
    current_time = 5000 # 5 seconds
    movement = f"{_movement}{stage}"
    for window in windows:
        start_time = current_time
        end_time = current_time + self.window_time_ms
        current_time = end_time
        print(f"Start time: {start_time}, End time: {end_time}, Intervals: {intervals[::2]}")
        in_movement = any(start_time >= interval_start * 1000 and end_time <= interval_end * 1000
                          for interval_start, interval_end in zip(intervals[::2], intervals[1::2]))
        movement_feature.append(in_movement)
    for i in range(len(windows)):
        window_features = self.features[stage][movement][column][i]
        window_features['InMovement'] = movement_feature[i]

  def save_extracted_features(self, patient, base_path='data/extracted_features'):
    def convert_to_serializable(obj):
      if isinstance(obj, np.integer):
        return int(obj)
      elif isinstance(obj, np.floating):
        return float(obj)
      elif isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
      elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
      else:
        return obj
            
    for stage in self.features.keys():
      print(f"Saving features for patient {patient} in stage {stage}")
      for movement in self.features[stage].keys():
        path = os.path.join(base_path, patient, stage)
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{movement}.json")
        with open(file_path, 'w') as f:
          json.dump(convert_to_serializable(self.features[stage][movement]), f)
        

  def __split_df(self, df, column_name, initial_time=None, final_time=None):
    if initial_time is None:
      return df[df[column_name] <= final_time]
    return df[(df[column_name] > initial_time) & (df[column_name] <= final_time)]

  def __windowing_data(self, data, window_time_ms, fs):
    window_size = int(window_time_ms * fs / 1000)
    data_size = len(data)
    windows = []
    for i in range(0, data_size, window_size):
      windows.append(data[i:i + window_size])
    return windows

  def __get_column_names(self):
    stage = list(self.movement_data.keys())[0]
    movement = list(self.movement_data[stage].keys())[0]

    df = self.movement_data[stage][movement]

    data_columns = [df.columns[i] for i in range(df.shape[1]) if i % 2 != 0]
    time_columns = [df.columns[i] for i in range(df.shape[1]) if i % 2 == 0]
    return data_columns, time_columns

  def __notch_filter(self, signal, fs=1926, Q=20, f0=60):
    b,a = iirnotch(f0, Q, fs)
    filtered = filtfilt(b, a, signal)
    return filtered
  
  def __bandpass_filter(self, signal, N = 4, freq_low = 20, freq_high = 450, fs = 1926):
    b,a = butter(N=N,Wn=[freq_low,freq_high],btype='bandpass',fs=fs)
    filtered = filtfilt(b,a,signal)
    return filtered

  def __extract_window_features(self, window):
    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    zc = self.__zero_crossings(window)
    ssc = self.__slope_sign_changes(window)
    wl = self.__waveform_length(window)
    median_freq, mean_freq = self.__frequency_features(window)
    return {
      'RMS': rms,
      'MAV': mav,
      'ZC': zc,
      'SSC': ssc,
      'WL': wl,
      'MedianFreq': median_freq,
      'MeanFreq': mean_freq
    }

  def __zero_crossings(self, window):
    return ((window[:-1] * window[1:]) < 0).sum()

  def __slope_sign_changes(self, window):
    diff = np.diff(window)
    return ((diff[:-1] * diff[1:]) < 0).sum()

  def __waveform_length(self, window):
    return np.sum(np.abs(np.diff(window)))

  def __frequency_features(self, window):
    freqs = np.fft.rfftfreq(len(window))
    fft_spectrum = np.abs(np.fft.rfft(window))
    fft_spectrum = fft_spectrum / np.sum(fft_spectrum)
    median_freq = np.median(freqs)
    mean_freq = np.sum(freqs * fft_spectrum)
    return median_freq, mean_freq
