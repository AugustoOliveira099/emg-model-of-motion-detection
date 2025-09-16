from utils import Patient
from SignalProcessorEMG import SignalProcessorEMG
from movement_intervals import movement_intervals_p1, movement_intervals_p4, movement_intervals_p5

def preprocess_and_save_signals():
  # Inicializa os pacientes
  # O parâmetro passado deve ser o nome da pasta que contém os dados do paciente
  patient1 = Patient("P1")
  patient2 = Patient("P4")
  patient3 = Patient("P5")

  # adiciona o cabeçalho aos dataframes
  patient1.add_header()
  patient2.add_header()
  patient3.add_header()

  # filtra as colunas relevantes
  patient1.get_relevant_columns()
  patient2.get_relevant_columns()
  patient3.get_relevant_columns()

  # Inicializa o processador de sinal
  signal_processor1 = SignalProcessorEMG(patient1.get_movement_signals())
  signal_processor2 = SignalProcessorEMG(patient2.get_movement_signals())
  signal_processor3 = SignalProcessorEMG(patient3.get_movement_signals())

  # Aplica filtros de frequência
  signal_processor1.apply_bandpass_filter()
  signal_processor1.apply_notch_filter()

  signal_processor2.apply_bandpass_filter()
  signal_processor2.apply_notch_filter()

  signal_processor3.apply_bandpass_filter()
  signal_processor3.apply_notch_filter()

  # Aplica retificação de onda completa
  signal_processor1.apply_full_wave_rectification()
  signal_processor2.apply_full_wave_rectification()
  signal_processor3.apply_full_wave_rectification()

  # Captura o segmento do sinal entre 5s e 25s
  signal_processor1.capture_signal_segment()
  signal_processor2.capture_signal_segment()
  signal_processor3.capture_signal_segment()

  # Aplica janelamento de 100ms aos dados
  signal_processor1.windowing_data(100)
  signal_processor2.windowing_data(100)
  signal_processor3.windowing_data(100)

  # Extrai as características dos sinais
  signal_processor1.extract_features(movement_intervals_p1)
  signal_processor2.extract_features(movement_intervals_p4)
  signal_processor3.extract_features(movement_intervals_p5)

  # Salva as características extraídas em um arquivo json
  signal_processor1.save_extracted_features(patient1.get_name())
  signal_processor2.save_extracted_features(patient2.get_name())
  signal_processor3.save_extracted_features(patient3.get_name())
