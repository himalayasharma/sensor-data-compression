import os
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(base_dir):

    logger = logging.getLogger(__name__)

    # -------------- Load statistics summary ----------------- 
    # Specify path to load processed data
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    with open(os.path.join(processed_data_dir, 'statistics_summary'), 'rb') as file_pi:
        statistics_summary = pickle.load(file_pi)
    logger.info('loaded statistics summary')
    statistics_summary['DR Algos'] = statistics_summary.index
    print(statistics_summary)

    # -------------- Load data dictionary ----------------- 
    with open(os.path.join(processed_data_dir, 'dr_data_dict_modelling'), 'rb') as file_pi:
        dr_data_dict_modelling = pickle.load(file_pi)
    logger.info('loaded data dictionary')

    # -------------- Load y_test ----------------- 
    with open(os.path.join(processed_data_dir, 'y_test'), 'rb') as file_pi:
        y_test = pickle.load(file_pi)

    # -------------- Size comparison plot -----------------
    ax1 = statistics_summary.iloc[1:, :].plot.bar(x='DR Algos', y='Compressed Size',
    rot=22.5, figsize=(8,8))
    ax1.set_yticks(np.arange(0, 180, step=20))
    ax1.bar_label(ax1.containers[0])
    ax1.set_title("Size of data after compression", fontsize=20)
    ax1.set_xlabel("Dimensionality reduction algorithms", fontsize=15)
    ax1.set_ylabel("Compressed size (KiB)", fontsize=15)
    plt.savefig('reports/figures/compressed_size_bar.png')
    logger.info('saved plot showing comparison of sizes') 

    # -------------- No of reduced dimensions plot -----------------
    ax2 = statistics_summary.iloc[1:, :].plot.bar(x='DR Algos', y='No of dims',
    rot=22.5, figsize=(8,8), color='brown')
    ax2.set_yticks(np.arange(0, 22))
    ax2.set_title("No of features after dimensionality reduction", fontsize=20)
    ax2.set_xlabel("Dimensionality reduction algorithms", fontsize=15)
    ax2.set_ylabel("No of features", fontsize=15)
    plt.savefig('reports/figures/no_of_features_bar.png')
    logger.info('saved plot showing number of features after dimensionality reduction')

    # -------------- Compression ratio & Space saving plot -----------------
    fig, ax3 = plt.subplots(2, 1, figsize=(10,8))
    statistics_summary.iloc[1:, :].plot.barh(x='DR Algos', y='Compression Ratio', rot=22.5, ax=ax3[0], color='orange')
    ax3[0].bar_label(ax3[0].containers[0])
    ax3[0].get_legend().remove()
    ax3[0].set_title("Compression ratio = Uncompressed size/Compressed size", fontsize=12)
    ax3[0].set_ylabel("Dimensionality reduction algorithms", fontsize=12)
    statistics_summary.iloc[1:, :].plot.barh(x='DR Algos', y='Space Saving', rot=22.5, ax=ax3[1], color='green')
    ax3[1].get_legend().remove()
    ax3[1].set_title("Space saving = 1 - (Compressed size/Uncompressed size)", fontsize=12)
    ax3[1].set_ylabel("Dimensionality reduction algorithms", fontsize=12)
    plt.savefig('reports/figures/compression_ratio_barh.png')
    logger.info('saved plot showing showing compression ratio & space saving')

    # -------------- Dimensionality reduction 2-d plots -----------------
    nrows, ncols = 2, 5
    fig, ax4 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,10))
    row = 0
    col = 0
    for key, item in dr_data_dict_modelling.items():
        ax4[row, col].scatter(item[2][:, 0], item[2][:, 1], c=y_test)
        ax4[row, col].set_title(key)
        col += 1
        if(col > 4):
            col = 0
            row = 1
    plt.savefig('reports/figures/dimensionality_reduction_plots.png')
    logger.info('saved plot showing showing data with 2 dimensions only')

    # -------------- Model accuracy plots -----------------
    print(statistics_summary.columns)
    ax5 = statistics_summary.iloc[1:, :].plot.bar(x='DR Algos', y=['KNN(n_neighbors=3) accuracy', 'Decision tree accuracy', 'Random forest accuracy', 'SVC accuracy'],
    rot=22.5, figsize=(8,8))
    ax5.set_title("Multi-class classification accuracy", fontsize=20)
    ax5.set_xlabel("Dimensionality reduction algorithms", fontsize=15)
    ax5.set_ylabel("Accuracy", fontsize=15)
    plt.savefig('reports/figures/model_accuracy_plots.png')
    logger.info('saved plot showing showing model accuracies')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(project_dir)