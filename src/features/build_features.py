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

def get_backward_elimination_data(X_train, X_test, y_train, y_test):

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.feature_selection import SequentialFeatureSelector
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    sfs_backward = SequentialFeatureSelector(knn, n_features_to_select=2, direction='backward', n_jobs=-1)
    sfs_backward.fit(X_train, y_train)
    return sfs_backward.transform(X_train), sfs_backward.transform(X_test)

def get_forward_selection_data(X_train, X_test, y_train, y_test):

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.feature_selection import SequentialFeatureSelector
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    sfs_forward = SequentialFeatureSelector(knn, n_features_to_select=2, direction='forward', n_jobs=-1)
    sfs_forward.fit(X_train, y_train)
    return sfs_forward.transform(X_train), sfs_forward.transform(X_test)

def get_random_forest_data(X_train, X_test, y_train, y_test):

    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    indices = np.argsort(random_forest.feature_importances_)[-2:]
    return X_train[:, indices], X_test[:, indices]

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
    logger.info('generated pca data')
    # ii) LDA
    dr_data_dict['LDA'] = get_lda_data(X_train, X_test, y_train, y_test)
    logger.info('generated lda data')
    # iii) Truncated SVD
    dr_data_dict['Truncated SVD'] = get_truncated_svd_data(X_train, X_test, y_train, y_test)
    logger.info('generated truncated svd data')

    # II. Non-linear methods
    # i) Kernel PCA
    dr_data_dict['Kernel PCA'] = get_kernel_pca_data(X_train, X_test, y_train, y_test)
    logger.info('generated kernel pca data')
    # ii) t-SNE
    dr_data_dict['tSNE'] = get_tsne_data(X_train, X_test, y_train, y_test)
    logger.info('generated t-sne data')
    # iii) MDS
    dr_data_dict['MDS'] = get_mds_data(X_train, X_test, y_train, y_test)
    logger.info('generated mds data')
    # iv) Isomap
    dr_data_dict['Isomap'] = get_isomap_data(X_train, X_test, y_train, y_test)
    logger.info('generated isomap data')

    # -------------- FEATURE SELECTION -----------------
    # Get PCA data with 10 features
    from sklearn.decomposition import PCA
    pca_feature_selection = PCA(n_components=10)
    pca_feature_selection.fit(X_train)
    X_train_pca = pca_feature_selection.transform(X_train)
    X_test_pca = pca_feature_selection.transform(X_test)
    
    # i) Backward elimination
    dr_data_dict['Backward Elimination'] = get_backward_elimination_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated backward elimination data')
    # ii) Forward selection
    dr_data_dict['Forward Selection'] = get_forward_selection_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated forward selection data')
    # iii) Random forest
    dr_data_dict['Random Forest'] = get_random_forest_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated random forest data')

    # -------------- Save data dictionary -----------------
    with open(os.path.join(processed_data_dir, 'dr_data_dict'), 'wb') as file_pi:
        pickle.dump(dr_data_dict, file_pi)
    logger.info('saved dictionary containing DR data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
