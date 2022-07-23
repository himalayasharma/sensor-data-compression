import os
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def get_arr_size(x):
    return (x.size*x.itemsize)/1024.0

def get_compression_ratio(input_arr, output_arr):
    return get_arr_size(input_arr)/get_arr_size(output_arr)

def get_space_saving(input_arr, output_arr):
    return 1-(get_arr_size(output_arr)/get_arr_size(input_arr))

def main(base_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # -------------- Load data dictionary -----------------
    logger = logging.getLogger(__name__)
    # Specify path to load processed data
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    # Load data dictionary
    with open(os.path.join(processed_data_dir, 'dr_data_dict_modelling'), 'rb') as file_pi:
        dr_data_dict = pickle.load(file_pi)
    logger.info('loaded dictionary containing data whose dimensions have been reduced')

    # -------------- Load X_train -----------------
    with open(os.path.join(processed_data_dir, 'X_train'), 'rb') as file_pi:
        X_train = pickle.load(file_pi)

   # -------------- Load X_test -----------------
    with open(os.path.join(processed_data_dir, 'X_test'), 'rb') as file_pi:
        X_test = pickle.load(file_pi)

    # -------------- Create statistic summary dataframe -----------------
    statistics_summary_list = list()
    statistics_summary = pd.DataFrame()

    # -------------- Add number of components -----------------
    n_components_dict = dict()
    n_components_dict['Original'] = 533
    for key, item in dr_data_dict.items():
        n_components_dict[key] = item[0]
    n_components = pd.DataFrame.from_dict(n_components_dict, orient='index', columns=['No of dims'])
    statistics_summary_list.append(n_components)

    # -------------- Calculate compressed sizes -----------------
    compressed_size_dict = dict()
    compressed_size_dict['Original'] = get_arr_size(X_test)
    for key, item in dr_data_dict.items():
        compressed_size_dict[key] = get_arr_size(item[2])
    compressed_size = pd.DataFrame.from_dict(compressed_size_dict, orient='index', columns=['Compressed Size'])
    statistics_summary_list.append(compressed_size)

    # -------------- Calculate compression ratios -----------------
    compression_ratio_dict = dict()
    compression_ratio_dict['Original'] = 1
    for key, item in dr_data_dict.items():
        compression_ratio_dict[key] = get_compression_ratio(X_test, item[2])
    compression_ratio = pd.DataFrame.from_dict(compression_ratio_dict, orient='index', columns=['Compression Ratio'])
    statistics_summary_list.append(compression_ratio)

    # -------------- Calculate space saving -----------------
    space_saving_dict = dict()
    space_saving_dict['Original'] = 0
    for key, item in dr_data_dict.items():
        space_saving_dict[key] = get_space_saving(X_test, item[2])
    space_saving = pd.DataFrame.from_dict(space_saving_dict, orient='index', columns=['Space Saving'])
    statistics_summary_list.append(space_saving)

    # -------------- Print statistics summary -----------------
    logger.info('created statistics summary')
    statistics_summary = pd.concat(statistics_summary_list, axis=1)
    print(statistics_summary)

    # -------------- Save statistics summary -----------------
    with open(os.path.join(processed_data_dir, 'statistics_summary'), 'wb') as file_pi:
        pickle.dump(statistics_summary, file_pi)
    logger.info('saved statistics summary')
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
