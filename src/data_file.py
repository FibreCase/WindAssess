import pandas as pd
import numpy as np
import os

def import_data(file_path):
    """
    Imports data from a CSV file and returns a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file to be imported.
    
    Returns:
    pd.DataFrame: A DataFrame containing the imported data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        data = pd.read_csv(file_path, parse_dates=["Time"])
        return data
    except Exception as e:
        print(f"An error occurred while importing the data: {e}")
        return None

def export_data(df, file_path):
    """
    Exports a pandas DataFrame to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be exported.
    file_path (str): The path where the CSV file will be saved.
    
    Returns:
    None
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Data successfully exported to {file_path}")
    except Exception as e:
        print(f"An error occurred while exporting the data: {e}")