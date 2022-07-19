import os
import logging
import pickle
from random import random
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_knn_accuracy(dr_data_dict, y_train, y_test):

    knn_acc_dict = dict()
    for key, item in dr_data_dict.items():
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(item[1], y_train)
        knn_acc_dict[key] = knn.score(item[2], y_test)
        knn_acc = pd.DataFrame.from_dict(knn_acc_dict, orient='index', columns=['KNN(n_neighbors=7) accuracy'])
    return knn_acc

def get_decision_tree_accuracy(dr_data_dict, y_train, y_test):

    decision_tree_acc_dict = dict()
    for key, item in dr_data_dict.items():
        decision_tree = DecisionTreeClassifier(random_state=42)
        decision_tree.fit(item[1], y_train)
        decision_tree_acc_dict[key] = decision_tree.score(item[2], y_test)
        decision_tree_acc = pd.DataFrame.from_dict(decision_tree_acc_dict, orient='index', columns=['Decision tree accuracy'])
    return decision_tree_acc

def get_random_forest_accuracy(dr_data_dict, y_train, y_test):

    random_forest_acc_dict = dict()
    for key, item in dr_data_dict.items():
        random_forest = RandomForestClassifier(random_state=42, n_jobs=-1)
        random_forest.fit(item[1], y_train)
        random_forest_acc_dict[key] = random_forest.score(item[2], y_test)
        random_forest_acc = pd.DataFrame.from_dict(random_forest_acc_dict, orient='index', columns=['Random forest accuracy'])
    return random_forest_acc

def get_svc_accuracy(dr_data_dict, y_train, y_test):

    svc_acc_dict = dict()
    for key, item in dr_data_dict.items():
        svc = LinearSVC(random_state=42)
        svc.fit(item[1], y_train)
        svc_acc_dict[key] = svc.score(item[2], y_test)
        svc_acc = pd.DataFrame.from_dict(svc_acc_dict, orient='index', columns=['SVC accuracy'])
    return svc_acc

def main(base_dir):

    logger = logging.getLogger(__name__)

    # -------------- Load train and test data -----------------
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

    # -------------- Load data dictionary -----------------
    with open(os.path.join(processed_data_dir, 'dr_data_dict_modelling'), 'rb') as file_pi:
        dr_data_dict = pickle.load(file_pi)
    # Add original data
    dr_data_dict["Original"] = (533, X_train, X_test)

    # -------------- Load statistics summary ----------------- 
    with open(os.path.join(processed_data_dir, 'statistics_summary'), 'rb') as file_pi:
        statistics_summary = pickle.load(file_pi)
    logger.info('loaded statistics summary')

    # -------------- Add KNN accuracy ----------------- 
    knn_acc = get_knn_accuracy(dr_data_dict, y_train, y_test)
    statistics_summary = pd.concat([statistics_summary, knn_acc], axis=1)
    logger.info('added knn accuracy to statistics summary')

   # -------------- Add Decision tree accuracy ----------------- 
    decision_tree_acc = get_decision_tree_accuracy(dr_data_dict, y_train, y_test)
    statistics_summary = pd.concat([statistics_summary, decision_tree_acc], axis=1)
    logger.info('added decision tree accuracy to statistics summary')

   # -------------- Add Random forest accuracy ----------------- 
    random_forest_acc = get_random_forest_accuracy(dr_data_dict, y_train, y_test)
    statistics_summary = pd.concat([statistics_summary, random_forest_acc], axis=1)
    logger.info('added random forest accuracy to statistics summary')

   # -------------- Add SVC accuracy ----------------- 
    svc_acc = get_svc_accuracy(dr_data_dict, y_train, y_test)
    statistics_summary = pd.concat([statistics_summary, svc_acc], axis=1)
    logger.info('added svc accuracy to statistics summary')

    print(statistics_summary)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    main(project_dir)