import numpy as np
import pandas as pd
from utils import Patient
from scipy.signal import iirnotch, filtfilt, butter

class SignalPreProcessor:
  def __init__(self, signal, name):
    self.patient = Patient(name)
    self.features = {}
    self.windowed_dict = {}
    self.header_file = "cabecalho"
    self.window_time_ms = 100
    self.signal = signal
    self.data_columns = self.time_columns = None


  def preprocess(self):
    self.__add_header()
    self.__get_relevant_columns()
    self.data_columns, self.time_columns = self.__get_column_names()
    self.__apply_bandpass_filter()
    self.__apply_notch_filter()
    self.__apply_full_wave_rectification()
    self.__capture_signal_segment()
    self.__apply_windowing_data()
    self.__extract_features()
    self.__features_to_df()
    self.__add_time_column()
    return self.features


  def get_signal(self):
    return self.signal
  

  def __add_header(self):
    self.patient.add_header()


  def __get_relevant_columns(self):
    self.patient.get_relevant_columns()


  def __get_column_names(self):
    data_columns = [self.signal.columns[i] for i in range(self.signal.shape[1]) if i % 2 != 0]
    time_columns = [self.signal.columns[i] for i in range(self.signal.shape[1]) if i % 2 == 0]
    return data_columns, time_columns
  

  def __apply_bandpass_filter(self, fs = 1926, lowcut = 20, highcut = 450, order = 4):
    def bandpass_filter(signal):
      b,a = butter(N=order,Wn=[lowcut,highcut],btype='bandpass',fs=fs)
      filtered = filtfilt(b, a, signal)
      return filtered
    
    for column in self.data_columns:
      self.signal[column] = bandpass_filter(self.signal[column])
  

  def __apply_notch_filter(self, fs = 1926, f0 = 60, Q = 70):
    def notch_filter(signal, fs=1926, Q=20, f0=60):
      b,a = iirnotch(f0, Q, fs)
      filtered = filtfilt(b, a, signal)
      return filtered
    
    for column in self.data_columns:
      self.signal[column] = notch_filter(self.signal[column], fs = fs, f0 = f0, Q = Q)

  
  def __apply_full_wave_rectification(self):
    for column in self.data_columns:
      self.signal[column] = np.abs(self.signal[column])


  def __capture_signal_segment(self, initial_time=5, final_time=25):
    def split_df(df, column_name):
      if initial_time is None:
        return df[df[column_name] <= final_time]
      return df[(df[column_name] > initial_time) & (df[column_name] <= final_time)]
  
    # As colunas de tempo possuem sempre os mesmo valores, escolhemos a primeira
    time_column = self.time_columns[0]
    self.signal = split_df(self.signal, time_column)
  

  def __apply_windowing_data(self, window_time_ms = 100, fs = 1926):
    def windowing_data(data):
      window_size = int(window_time_ms * fs / 1000)
      data_size = len(data)
      windows = []
      for i in range(0, data_size, window_size):
        windows.append(data[i:i + window_size])
      return windows
    
    self.window_time_ms = window_time_ms
    windowed_dict = {}
    for column in self.data_columns:
      windowed_data = windowing_data(self.signal[column])
      windowed_dict[column] = windowed_data
    self.windowed_dict = windowed_dict


  def __extract_features(self):
    for column in self.data_columns:
      windows = self.windowed_dict[column]
      self.features[column] = [self.__extract_window_features(window) for window in windows]


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


  def __features_to_df(self):
    for column in self.data_columns:
      features = self.features[column]
      df = pd.DataFrame(features)
      self.features[column] = df

  
  def __add_time_column(self):
    for column in self.data_columns:
      self.features[column]['Time'] = np.linspace(5, 25, len(self.features[column]), endpoint=False)