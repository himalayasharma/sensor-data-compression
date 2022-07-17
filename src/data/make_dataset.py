# -*- coding: utf-8 -*-
import os, wget, zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def download_data(destination_dir, url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6960825/bin/sensors-19-05524-s001.zip"):

    # Make path if does not exits
    if(os.path.exists(destination_dir) == False):
        os.makedirs(destination_dir)
    # Download data if it does not exist on disk
    if(os.path.exists(os.path.join(destination_dir, "sensors-19-05524-s001.zip")) == False):
        wget.download(url, destination_dir)
        
def extract_data(data_dir, filename="sensors-19-05524-s001.zip"):
    
    with zipfile.ZipFile(os.path.join(data_dir, filename), 'r') as zip_ref:
        zip_ref.extractall(data_dir)


def main(base_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
  # -------------- Download and extract data -----------------
    logger = logging.getLogger(__name__)
    # Specify path to which data that will be downloaded
    raw_data_dir = os.path.join(base_dir, 'data/raw')
    # Download raw data
    logger.info(f'downloading physiological sensor data to {raw_data_dir}')
    download_data(raw_data_dir, url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6960825/bin/sensors-19-05524-s001.zip")
    logger.info('downloaded data')
    # Extract raw data
    logger.info(f'starting extraction of data to {raw_data_dir}')
    extract_data(raw_data_dir)
    logger.info('extracted data')

    # -------------- Load data -----------------
    # Load data
    data_path = os.path.join(raw_data_dir, 'data.txt')
    data = pd.read_csv(data_path, header=None)

    # Load column headers
    column_headers_path = os.path.join(raw_data_dir, 'labels.txt')
    column_headers = np.genfromtxt(column_headers_path, delimiter='\n', dtype='str')

    # Insert column headers
    data.columns = column_headers
    logger.info('loaded data')

    # -------------- Data wrangling -----------------
    # Get input data
    X = data.iloc[:, :-1]
    X = np.array(X)

    # Get target
    y = np.array(data.iloc[:, -1])
    # Conversion process - '\x01.0f' ---> replace('.0f', '') ---> '\x01' ---> ord() ---> 1
    y = np.array(list(map(lambda x:ord(x.replace('.0f', '')), y))).astype('float')

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # -------------- Save preprocessed train & test sets -----------------
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    if(os.path.exists(processed_data_dir) == False):
        os.makedirs(processed_data_dir)

    with open(os.path.join(processed_data_dir, 'X_train'), 'wb') as file_pi:
        pickle.dump(X_train, file_pi)
    with open(os.path.join(processed_data_dir, 'y_train'), 'wb') as file_pi:
        pickle.dump(y_train, file_pi)
    with open(os.path.join(processed_data_dir, 'X_test'), 'wb') as file_pi:
        pickle.dump(X_test, file_pi)
    with open(os.path.join(processed_data_dir, 'y_test'), 'wb') as file_pi:
        pickle.dump(y_test, file_pi)
    logger.info(f'saved X and y to {processed_data_dir}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
