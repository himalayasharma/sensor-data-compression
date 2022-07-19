# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

def get_2d_pca_data(X_train, X_test, y_train, y_test):

    pca = PCA(n_components=2)
    pca.fit(X_train)
    return pca.transform(X_train), pca.transform(X_test)

def get_pca_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=5)
        pca = PCA(n_components=n_components)
        pca.fit(X_train, y_train)
        X_train_pca, X_test_pca = pca.transform(X_train), pca.transform(X_test)
        knn.fit(X_train_pca, y_train)
        if knn.score(X_test_pca, y_test) > 0.8:
            return n_components, X_train_pca, X_test_pca
    return n_components, X_train_pca, X_test_pca

def get_2d_lda_data(X_train, X_test, y_train, y_test):

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X_train, y_train)
    return lda.transform(X_train), lda.transform(X_test)

def get_lda_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(1, 4):
        knn = KNeighborsClassifier(n_neighbors=5)
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X_train, y_train)
        X_train_lda, X_test_lda = lda.transform(X_train), lda.transform(X_test)
        knn.fit(X_train_lda, y_train)
        if knn.score(X_test_lda, y_test) > 0.8:
            return n_components, X_train_lda, X_test_lda
    return n_components, X_train_lda, X_test_lda

def get_2d_truncated_svd_data(X_train, X_test, y_train, y_test):

    truncated_svd = TruncatedSVD(n_components=2)
    truncated_svd.fit(X_train)
    return truncated_svd.transform(X_train), truncated_svd.transform(X_test)

def get_truncated_svd_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=5)
        truncated_svd = TruncatedSVD(n_components=n_components)
        truncated_svd.fit(X_train, y_train)
        X_train_truncated_svd, X_test_truncated_svd = truncated_svd.transform(X_train), truncated_svd.transform(X_test)
        knn.fit(X_train_truncated_svd, y_train)
        if knn.score(X_test_truncated_svd, y_test) > 0.8:
            return n_components, X_train_truncated_svd, X_test_truncated_svd
    return n_components, X_train_truncated_svd, X_test_truncated_svd

def get_2d_kernel_pca_data(X_train, X_test, y_train, y_test):

    kernel_pca = KernelPCA(n_components=2, n_jobs=-1)
    kernel_pca.fit(X_train)
    return kernel_pca.transform(X_train), kernel_pca.transform(X_test)

def get_kernel_pca_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=5)
        kernel_pca = KernelPCA(n_components=n_components)
        kernel_pca.fit(X_train, y_train)
        X_train_kernel_pca, X_test_kernel_pca = kernel_pca.transform(X_train), kernel_pca.transform(X_test)
        knn.fit(X_train_kernel_pca, y_train)
        if knn.score(X_test_kernel_pca, y_test) > 0.8:
            return n_components, X_train_kernel_pca, X_test_kernel_pca
    return n_components, X_train_kernel_pca, X_test_kernel_pca

def get_2d_tsne_data(X_train, X_test, y_train, y_test):

    tsne = TSNE(n_components=2, n_jobs=-1)
    return tsne.fit_transform(X_train).astype('float'), tsne.fit_transform(X_test).astype('float')

def get_tsne_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(1, 4):
        knn = KNeighborsClassifier(n_neighbors=5)
        tsne = TSNE(n_components=n_components, n_jobs=-1)
        tsne.fit(X_train, y_train)
        X_train_tsne, X_test_tsne = tsne.fit_transform(X_train).astype('float'), tsne.fit_transform(X_test).astype('float')
        knn.fit(X_train_tsne, y_train)
        if knn.score(X_test_tsne, y_test) > 0.8:
            return n_components, X_train_tsne, X_test_tsne
    return n_components, X_train_tsne, X_test_tsne

def get_2d_mds_data(X_train, X_test, y_train, y_test):

    mds = MDS(n_components=2, n_jobs=-1)
    return mds.fit_transform(X_train), mds.fit_transform(X_test)

def get_mds_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(18, 21):
        knn = KNeighborsClassifier(n_neighbors=5)
        mds = MDS(n_components=n_components, n_jobs=-1)
        mds.fit(X_train, y_train)
        X_train_mds, X_test_mds = mds.fit_transform(X_train), mds.fit_transform(X_test)
        knn.fit(X_train_mds, y_train)
        if knn.score(X_test_mds, y_test) > 0.8:
            return n_components, X_train_mds, X_test_mds
    return n_components, X_train_mds, X_test_mds

def get_2d_isomap_data(X_train, X_test, y_train, y_test):

    isomap = Isomap(n_components=2, n_jobs=-1)
    return isomap.fit_transform(X_train), isomap.fit_transform(X_test)

def get_isomap_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(18, 21):
        knn = KNeighborsClassifier(n_neighbors=5)
        isomap = Isomap(n_components=n_components, n_jobs=-1)
        isomap.fit(X_train, y_train)
        X_train_isomap, X_test_isomap = isomap.fit_transform(X_train), isomap.fit_transform(X_test)
        knn.fit(X_train_isomap, y_train)
        if knn.score(X_test_isomap, y_test) > 0.8:
            return n_components, X_train_isomap, X_test_isomap
    return n_components, X_train_isomap, X_test_isomap

def get_2d_backward_elimination_data(X_train, X_test, y_train, y_test):
    
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    sfs_backward = SequentialFeatureSelector(knn, n_features_to_select=2, direction='backward', n_jobs=-1)
    sfs_backward.fit(X_train, y_train)
    return sfs_backward.transform(X_train), sfs_backward.transform(X_test)

def get_backward_elimination_data(X_train, X_test, y_train, y_test):
    
    for n_components in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=5)
        sfs_backward = SequentialFeatureSelector(knn, n_features_to_select=n_components, direction='backward', n_jobs=-1)
        sfs_backward.fit(X_train, y_train)
        X_train_backward_elim, X_test_backward_elim = sfs_backward.transform(X_train), sfs_backward.transform(X_test)
        knn.fit(X_train_backward_elim, y_train)
        if knn.score(X_test_backward_elim, y_test) > 0.8:
            return n_components, X_train_backward_elim, X_test_backward_elim
    return n_components, X_train_backward_elim, X_test_backward_elim

def get_2d_forward_selection_data(X_train, X_test, y_train, y_test):

    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    sfs_forward = SequentialFeatureSelector(knn, n_features_to_select=2, direction='forward', n_jobs=-1)
    sfs_forward.fit(X_train, y_train)
    return sfs_forward.transform(X_train), sfs_forward.transform(X_test)

def get_forward_selection_data(X_train, X_test, y_train, y_test):

    for n_components in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=5)
        sfs_forward = SequentialFeatureSelector(knn, n_features_to_select=n_components, direction='forward', n_jobs=-1)
        sfs_forward.fit(X_train, y_train)
        X_train_forward_select, X_test_forward_select = sfs_forward.transform(X_train), sfs_forward.transform(X_test)
        knn.fit(X_train_forward_select, y_train)
        if knn.score(X_test_forward_select, y_test) > 0.8:
            return n_components, X_train_forward_select, X_test_forward_select
    return n_components, X_train_forward_select, X_test_forward_select

def get_2d_random_forest_data(X_train, X_test, y_train, y_test):

    random_forest = RandomForestClassifier(n_jobs=-1)
    random_forest.fit(X_train, y_train)
    indices = np.argsort(random_forest.feature_importances_)[-2:]
    return X_train[:, indices], X_test[:, indices]

def get_random_forest_data(X_train, X_test, y_train, y_test):

    for n_components in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=5)
        random_forest = RandomForestClassifier(n_jobs=-1)
        random_forest.fit(X_train, y_train)
        indices = np.argsort(random_forest.feature_importances_)[-n_components:]
        X_train_random_forest, X_test_random_forest = X_train[:, indices], X_test[:, indices]
        knn.fit(X_train_random_forest, y_train)
        if knn.score(X_test_random_forest, y_test) > 0.8:
            return n_components, X_train_random_forest, X_test_random_forest
    return n_components, X_train_random_forest, X_test_random_forest

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
    dr_data_dict_modelling = dict()

    # -------------- FEATURE EXTRACTION -----------------
    # I. Linear methods
    # i) PCA
    dr_data_dict['PCA'] = get_2d_pca_data(X_train, X_test, y_train, y_test)
    logger.info('generated 2d pca data')
    dr_data_dict_modelling['PCA'] = get_pca_data(X_train, X_test, y_train, y_test)
    logger.info('generated pca data')
    # ii) LDA
    dr_data_dict['LDA'] = get_2d_lda_data(X_train, X_test, y_train, y_test)
    logger.info('generated 2d lda data')
    dr_data_dict_modelling['LDA'] = get_lda_data(X_train, X_test, y_train, y_test)
    logger.info('generated lda data')
    # iii) Truncated SVD
    dr_data_dict['Truncated SVD'] = get_2d_truncated_svd_data(X_train, X_test, y_train, y_test)
    logger.info('generated 2d truncated svd data')
    dr_data_dict_modelling['Truncated SVD'] = get_truncated_svd_data(X_train, X_test, y_train, y_test)
    logger.info('generated truncated svd data')

    # II. Non-linear methods
    # i) Kernel PCA
    dr_data_dict['Kernel PCA'] = get_2d_kernel_pca_data(X_train, X_test, y_train, y_test)
    logger.info('generated 2d kernel pca data')
    dr_data_dict_modelling['Kernel PCA'] = get_kernel_pca_data(X_train, X_test, y_train, y_test)
    logger.info('generated kernel pca data')
    # ii) t-SNE
    dr_data_dict['tSNE'] = get_2d_tsne_data(X_train, X_test, y_train, y_test)
    logger.info('generated 2d t-sne data')
    dr_data_dict_modelling['tSNE'] = get_tsne_data(X_train, X_test, y_train, y_test)
    logger.info('generated t-sne data')
    # iii) MDS
    dr_data_dict['MDS'] = get_2d_mds_data(X_train, X_test, y_train, y_test)
    logger.info('generated 2d mds data')
    dr_data_dict_modelling['MDS'] = get_mds_data(X_train, X_test, y_train, y_test)
    logger.info('generated mds data')
    # iv) Isomap
    dr_data_dict['Isomap'] = get_2d_isomap_data(X_train, X_test, y_train, y_test)
    logger.info('generated 2d isomap data')
    dr_data_dict_modelling['Isomap'] = get_isomap_data(X_train, X_test, y_train, y_test)
    logger.info('generated isomap data')

    # -------------- FEATURE SELECTION -----------------
    # Get PCA data with 10 features
    from sklearn.decomposition import PCA
    pca_feature_selection = PCA(n_components=10)
    pca_feature_selection.fit(X_train)
    X_train_pca = pca_feature_selection.transform(X_train)
    X_test_pca = pca_feature_selection.transform(X_test)
    
    # i) Backward elimination
    dr_data_dict['Backward Elimination'] = get_2d_backward_elimination_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated 2d backward elimination data')
    dr_data_dict_modelling['Backward Elimination'] = get_backward_elimination_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated backward elimination data')
    # ii) Forward selection
    dr_data_dict['Forward Selection'] = get_2d_forward_selection_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated 2d forward selection data')
    dr_data_dict_modelling['Forward Selection'] = get_forward_selection_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated forward selection data')
    # iii) Random forest
    dr_data_dict['Random Forest'] = get_2d_random_forest_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated 2d random forest data')
    dr_data_dict_modelling['Random Forest'] = get_random_forest_data(X_train_pca, X_test_pca, y_train, y_test)
    logger.info('generated random forest data')

    # -------------- Save data dictionary -----------------'
    with open(os.path.join(processed_data_dir, 'dr_data_dict'), 'wb') as file_pi:
        pickle.dump(dr_data_dict, file_pi)
    logger.info('saved dictionary containing data whose dimensions have been reduced for plotting')

    with open(os.path.join(processed_data_dir, 'dr_data_dict_modelling'), 'wb') as file_pi:
        pickle.dump(dr_data_dict_modelling, file_pi)
    logger.info('saved dictionary containing data whose dimensions have been reduced')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
