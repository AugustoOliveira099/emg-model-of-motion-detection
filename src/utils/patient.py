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
        print("Filtering relevant columns (idempotent)")
        # Itera sobre os estágios
        stages = os.listdir(self.data_path)
        for stage in stages:
            # Percorre todos os dados
            for file_name in self.movements[stage].keys():
                df = self.movements[stage][file_name]
                original_shape = df.shape
                changed = False

                # 1) Remover colunas de acelerômetro se existirem ('.X', '.Y', '.Z')
                accel_cols = [c for c in df.columns if str(c).endswith(
                    '.X') or str(c).endswith('.Y') or str(c).endswith('.Z')]
                if accel_cols:
                    # Use filter with regex to remove accel cols (same behavior as antes)
                    df = df.filter(regex=r'^(?:(?!\.X$|\.Y$|\.Z$).)*$', axis=1)
                    changed = True
                    print(
                        f"Removed accel columns from {file_name}: {len(accel_cols)} columns removed")
                else:
                    print(
                        f"No accel columns found in {file_name}; skipping accel removal")

                # 2) Manter apenas as colunas numeradas de 3 a 10, se existirem
                # Detect columns that end with a number token
                numbered_cols = [c for c in df.columns if str(
                    c).split()[-1].isdigit()]
                cols_to_keep = [c for c in numbered_cols if 3 <=
                                int(str(c).split()[-1]) <= 10]
                if cols_to_keep:
                    # If the current columns already equal the desired set, skip
                    if list(df.columns) != cols_to_keep:
                        df = df[cols_to_keep]
                        changed = True
                        print(
                            f"Selected numbered columns 3-10 for {file_name}: kept {len(cols_to_keep)} columns")
                    else:
                        print(
                            f"Numbered columns 3-10 already selected for {file_name}; skipping")
                else:
                    print(
                        f"No numbered columns 3-10 found in {file_name}; skipping column selection")

                # 3) Truncar os dados acima de 25s apenas se a primeira coluna for numérica
                try:
                    times = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                    if times.notna().any():
                        mask = times <= 25
                        new_df = df[mask]
                        if new_df.shape != df.shape:
                            df = new_df.reset_index(drop=True)
                            changed = True
                            print(
                                f"Truncated {file_name} to <=25s: {original_shape[0] - df.shape[0]} rows removed")
                        else:
                            print(
                                f"{file_name} already truncated to <=25s; skipping")
                    else:
                        print(
                            f"First column in {file_name} is not numeric; skipping time truncation")
                except Exception as e:
                    print(
                        f"Error while truncating {file_name}: {e}; skipping truncation")

                # Save only if something changed
                if changed:
                    self.movements[stage][file_name] = df
                    df.to_csv(
                        f"{self.data_path}/{stage}/{file_name}.csv", index=False)
                else:
                    print(f"No changes for {file_name}; file not rewritten")

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

            # Add header to each dataframe only when necessary
            stages = os.listdir(self.data_path)
            for stage in stages:
                for file_name in list(self.movements[stage].keys()):
                    df = self.movements[stage][file_name]
                    # Normalize current columns to strings for comparison
                    current_cols = [str(c) for c in df.columns]

                    # Case 1: file already has the correct header -> skip
                    if current_cols == formated_header:
                        print(f"Skipping {file_name}: header already present")
                        continue

                    # Case 2: header is present as the first row (happens when header was added to data instead
                    # of as column names). Detect and fix by dropping the first row and assigning columns.
                    try:
                        first_row = [str(x).strip()
                                     for x in df.iloc[0].tolist()]
                    except Exception:
                        first_row = []

                    if first_row == formated_header:
                        print(
                            f"Fixing {file_name}: detected header in first row -> dropping it and assigning proper columns")
                        # Drop the first row and reset index, then set columns
                        fixed_df = df.iloc[1:].reset_index(drop=True)
                        fixed_df.columns = formated_header
                        self.movements[stage][file_name] = fixed_df
                        fixed_df.to_csv(
                            f"{self.data_path}/{stage}/{file_name}.csv", index=False)
                        continue

                    # Case 3: If number of columns matches expected header length, simply assign the header.
                    if len(df.columns) == len(formated_header):
                        print(
                            f"Adding header to {file_name}: assigning {len(formated_header)} column names")
                        df.columns = formated_header
                        self.movements[stage][file_name] = df
                        df.to_csv(
                            f"{self.data_path}/{stage}/{file_name}.csv", index=False)
                        continue

                    # Otherwise, columns don't match header length: warn and skip to avoid corrupting data
                    print(
                        f"Warning: cannot add header to {file_name} (columns: {len(df.columns)} != expected: {len(formated_header)}). Skipping file.")

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
            pattern = re.compile(
                r'Label: (.*?)(?:\s*Sampling frequency:|$)', re.DOTALL)
            match = pattern.search(label)

            # Check if there was a match and get the extracted value
            if match:
                label_value = match.group(1).strip()
                formated_header.append(f"TIME OF {label_value}")
                formated_header.append(label_value)
            else:
                print("Could not find the value between Label and Sampling frequency.")

        return formated_header
