import os
import re
import pandas as pd

class Patient:
  def __init__(self, name):
    """
    Initialize a Patient object.

    Args:
      name (str): The name of the patient.
    """
    self.name = name
    self.header_file = "cabecalho"
    root_path_data = "./data/patients"  # Define the base path
    self.data_path = f"{root_path_data}/{name}"
    self.movements = self.__get_movement_data()

  def get_name(self):
    return self.name

  def __get_movement_data(self):
    """
    Retrieve movement data for the patient.

    Returns:
      dict: A dictionary containing movement data for each stage.
    """
    stages = os.listdir(self.data_path)
    movement_files = {}
    for stage in stages:
      print(f"Processing stage {stage} for patient {self.name}")
      movement_files_list = os.listdir(f"{self.data_path}/{stage}")
      for file in movement_files_list:
        print(f"Processing file {file}")
        df = pd.read_csv(f"{self.data_path}/{stage}/{file}")
        file_name = file.split(".")[0]
        if movement_files.get(stage) is None:
          movement_files[stage] = {}
        movement_files[stage][file_name] = df

    return movement_files
  
  def get_movement_signals(self):
    return self.movements
  
  def get_relevant_columns(self):
    """
    Processes the movement data for each stage by performing the following steps:
    
    1. Iterates over the stages in the data path.
    2. For each file in the movements dictionary of the current stage:
       - Excludes columns that contain '.X', '.Y', and '.Z' (accelerometer data).
       - Retains columns numbered from 3 to 10.
       - Truncates the data to include only rows where the first column's value is less than or equal to 25 seconds.
       - Saves the processed data back to a CSV file in the corresponding stage directory.
    
    Note:
    - The method assumes that the first column contains time data in seconds.
    - The method modifies the `self.movements` dictionary in place and saves the processed data to CSV files.
    - This method should be called only once, as it processes and saves the data.
    """
    print ("Filtering relevant columns")
    # Itera sobre os estágios
    stages = os.listdir(self.data_path)
    for stage in stages:
      # Percorre todos os dados
      for file_name in self.movements[stage].keys():
        # Exclui as colunas que contêm '.X', '.Y' e '.Z' (dados do acelerômetro)
        self.movements[stage][file_name] = self.movements[stage][file_name].filter(regex='^(?:(?!\.X$|\.Y$|\.Z$).)*$', axis=1)

        # Mantem as conlunas de 3 a 10
        columns_to_maintain = [col for col in self.movements[stage][file_name].columns if col[-1].isdigit() and 3 <= int(col.split()[-1]) <= 10]
        self.movements[stage][file_name] = self.movements[stage][file_name][columns_to_maintain]

        # Trunca os dados acima dos 25s
        self.movements[stage][file_name] = self.movements[stage][file_name][self.movements[stage][file_name].iloc[:, 0] <= 25]

        self.movements[stage][file_name].to_csv(f"{self.data_path}/{stage}/{file_name}.csv", index=False)

  
  def add_header(self):
    """
    Add headers to the movement data files.

    This method should be called only for newly added data, as it adds the header to the CSV files.
    If the headers are already present in the files, calling this method again is unnecessary.
    """
    # Read the header file
    with open(f"data/{self.header_file}.csv", 'r') as file:
      header_lines = file.readlines()
      header = [linha.strip() for linha in header_lines]

      # Define the labels for the dataframe
      formated_header = self.__define_labels(header)

      # Add header to each dataframe
      stages = os.listdir(self.data_path)
      for stage in stages:
        for file_name in self.movements[stage].keys():
          print(f"Adding header to file {file_name}")
          self.movements[stage][file_name] = pd.DataFrame(self.movements[stage][file_name].values, columns=formated_header)
          self.movements[stage][file_name].to_csv(f"{self.data_path}/{stage}/{file_name}.csv", index=False)

  def __define_labels(self, header):
    """
    Define labels for the dataframe based on the header file.

    Args:
      header (list): A list of header lines.

    Returns:
      list: A list of formatted header labels.
    """
    formated_header = []

    for label in header:
      pattern = re.compile(r'Label: (.*?)(?:\s*Sampling frequency:|$)', re.DOTALL)
      match = pattern.search(label)

      # Check if there was a match and get the extracted value
      if match:
          label_value = match.group(1).strip()
          formated_header.append(f"TIME OF {label_value}")
          formated_header.append(label_value)
      else:
          print("Could not find the value between Label and Sampling frequency.")

    return formated_header
