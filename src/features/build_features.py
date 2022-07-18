# -*- coding: utf-8 -*-
import os, wget, zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def get_pca_data(X_train, X_test, y_train, y_test):

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X_train)
    return pca.transform(X_train), pca.transform(X_test)

def get_lda_data(X_train, X_test, y_train, y_test):

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    return lda.transform(X_train), lda.transform(X_test)

def get_truncated_svd_data(X_train, X_test, y_train, y_test):

    from sklearn.decomposition import TruncatedSVD
    truncated_svd = TruncatedSVD(n_components=2)
    truncated_svd.fit(X_train)
    return truncated_svd.transform(X_train), truncated_svd.transform(X_test)

def get_kernel_pca_data(X_train, X_test, y_train, y_test):

    from sklearn.decomposition import KernelPCA
    kernel_pca = KernelPCA(n_components=2, n_jobs=-1)
    kernel_pca.fit(X_train)
    return kernel_pca.transform(X_train), kernel_pca.transform(X_test)

def get_tsne_data(X_train, X_test, y_train, y_test):

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, n_jobs=-1)
    return tsne.fit_transform(X_train), tsne.fit_transform(X_test)

def get_mds_data(X_train, X_test, y_train, y_test):

    from sklearn.manifold import MDS
    mds = MDS(n_components=2, n_jobs=-1)
    return mds.fit_transform(X_train), mds.fit_transform(X_test) 

def get_isomap_data(X_train, X_test, y_train, y_test):

    from sklearn.manifold import MDS
    mds = MDS(n_components=2, n_jobs=-1)
    return mds.fit_transform(X_train), mds.fit_transform(X_test)

def main(base_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
  # -------------- Load train and test data -----------------
    logger = logging.getLogger(__name__)
    # Specify path to load processed data
    processed_data_dir = os.path.join(base_dir, 'data/processed')
    # Load train data
    with open(os.path.join(processed_data_dir, 'X_train'), 'rb') as file_pi:
        X_train = pickle.load(file_pi)
    with open(os.path.join(processed_data_dir, 'y_train'), 'rb') as file_pi:
        y_train = pickle.load(file_pi)
    logger.info('loaded train data')
    # Load test data
    with open(os.path.join(processed_data_dir, 'X_test'), 'rb') as file_pi:
        X_test = pickle.load(file_pi)
    with open(os.path.join(processed_data_dir, 'y_test'), 'rb') as file_pi:
        y_test = pickle.load(file_pi)
    logger.info('loaded test data')
    
    # -------------- Create dictionary to store DR data -----------------
    dr_data_dict = dict()

    # -------------- FEATURE EXTRACTION -----------------
    # I. Linear methods
    # i) PCA
    dr_data_dict['PCA'] = get_pca_data(X_train, X_test, y_train, y_test)
    # ii) LDA
    dr_data_dict['LDA'] = get_lda_data(X_train, X_test, y_train, y_test)
    # iii) Truncated SVD
    dr_data_dict['Truncated SVD'] = get_truncated_svd_data(X_train, X_test, y_train, y_test)

    # II. Non-linear methods
    # i) Kernel PCA
    dr_data_dict['Kernel PCA'] = get_kernel_pca_data(X_train, X_test, y_train, y_test)
    # ii) t-SNE
    dr_data_dict['tSNE'] = get_tsne_data(X_train, X_test, y_train, y_test)
    # iii) MDS
    dr_data_dict['MDS'] = get_mds_data(X_train, X_test, y_train, y_test)
    # iv) Isomap
    dr_data_dict['Isomap'] = get_isomap_data(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
