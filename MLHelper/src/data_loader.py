import pandas as pd

class DataLoader:
    """A class to load and manage datasets."""

    def __init__(self):
        self.data = None

    def load_csv(self, filepath, sep=',', usecols=None, names=None, index_col=None, dtype=None):
        """
        Loads a CSV file into a Pandas DataFrame.

        Parameters:
            - filepath (str): Path to the CSV file.
            - sep (str): Delimiter (default: ',').
            - usecols (list): Columns to load (default: None, loads all).
            - names (list): Column names (default: None).
            - index_col (int/str): Column to use as index (default: None).
            - dtype (dict): Data types for columns (default: None).

        Returns:
            - pd.DataFrame: Loaded dataset.
        """
        if not filepath.endswith('.csv'):
            raise ValueError("File should be in CSV format.")

        try:
            self.data = pd.read_csv(
                filepath,
                sep=sep,
                names=names,
                index_col=index_col,
                usecols=usecols,
                dtype=dtype
            )
            print(f"CSV file loaded successfully! Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            raise RuntimeError(f"Error loading CSV: {e}")

    def load_excel(self, filepath, sheet_name=0, usecols=None, index_col=None, dtype=None):
        """
        Loads an Excel file into a Pandas DataFrame.

        Parameters:
            - filepath (str): Path to the Excel file.
            - sheet_name (str/int): Sheet name or index (default: 0).
            - usecols (list): Columns to load (default: None, loads all).
            - index_col (int/str): Column to use as index (default: None).
            - dtype (dict): Data types for columns (default: None).

        Returns:
            - pd.DataFrame: Loaded dataset.
        """
        if not filepath.endswith('.xlsx'):
            raise ValueError("File should be in Excel (.xlsx) format.")

        try:
            self.data = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                usecols=usecols,
                index_col=index_col,
                dtype=dtype
            )
            print(f"Excel file loaded successfully! Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            raise RuntimeError(f"Error loading Excel file: {e}")

    def get_data(self):
        """
        Returns the loaded dataset.
        """
        if self.data is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        return self.data