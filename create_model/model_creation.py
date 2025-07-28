'''
XGBoost model for song classification.

This module provides functions from training and evaluation an XGBoost
classifer on song data, including visualization of results.
'''

import logging
import os
import pickle
import time

# IMPORTANT: Configure environment before scikit-learn import
# These settings affect joblib's multiprocessing behavior
os.environ['JOBLIB_VERBOSITY'] = '100'   # Enable verbose joblib output
os.environ['LOKY_PICKLER'] = 'pickle'    # Use pickle for serialization
os.environ['PYTHONUNBUFFERED'] = '1'     # Disable output buffering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
)
from sklearn.model_selection import (
        StratifiedKFold,
        train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, plot_importance

from create_model.models import RandomizedSearchCVWithProgress
from file_processing.create_clip_embeddings import get_feature_labels
from file_processing.logger import create_logger

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Embedding Types
VCE = 'vocal_clip_embeddings'
ICE = 'intrumental_clip_embeddings'

# Model Hyperparameters RandomizedSearchCVWithProgress
FEATURE_LABELS = get_feature_labels() # labels for librosa features
RANDOM_STATE = 42 # random state; used for reproducibility
# create a 75-15-15 train-eval-test set
TEST_SIZE = 0.15 # proportion of data used for test set
EVALUATION_SIZE = 15.0/85.0 # proportion of training data used for evaluation
N_JOBS = 4 # number of threads to create; used for multi-threading
PARAM_DISTRIBUTIONS = { # used for RandomSearchCV
  # Tree parameters
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 6),
    # Boosting parameters
    'n_estimators': randint(100, 400),
    'learning_rate': uniform(0.01, 0.19),
    # Regularization for many features
    'colsample_bytree': uniform(0.3, 0.5),
    'subsample': uniform(0.7, 0.25),
    # L1/L2 regularization
    'reg_alpha': uniform(0, 2),
    'reg_lambda': uniform(0.5, 2)
    }

# Default File Paths
CONFUSION_MATRIX_PATH = 'figures/confusion_matrix.png' 
FEATURE_IMPORTANCE_PATH = 'figures/feature_importance.png'
XGBOOST_TREE_PATH = 'figures/xgboost_tree.png'

# Debug
GET_REMOVED_SONGS = False # used to see list of songs removed by the keyword filter
ADDITIONAL_BANNED_WORDS = ['mix', '~', 'kr', 'jp']




# ===========================================================================
# FUNCTIONS
# ===========================================================================

'''
[Helper function for create_train_eval_test_split] Given an embedding column of
a DataFrame, returns a list of all the clip embeddings in that column.
'''
def get_embeddings(df):
    return [np.abs(clip_embedding) for clip_embeddings in df for clip_embedding
            in clip_embeddings]
'''
[Helper function for create_train_eval_test_split] Given a list of song labels
and an embeddong column of a DataFrame, returns a list of song labels to
correspond to all of the clips.
'''
def make_labels(song_labels, df):
    return [song_labels[i] for i in range(len(song_labels)) for j in
            range(len(df.iloc[i]))]

'''
Given a DataFrame of Song object metadata, runs train_test_split on the rows of
the df and creates a usable X_train, X_eval, X_test, y_train, y_eval, y_test
sets for run_xgb. Will get the embeddings for X from embed_types and choice of
y by label_name.
'''
def create_train_eval_test_split(
        input_df,
        label_df,
        embed_types=[VCE],
        test_size=TEST_SIZE,
        evaluation_size=EVALUATION_SIZE,
        random_state=RANDOM_STATE,
        ):
    Song_train, Song_test, label_train, label_test = train_test_split(
            input_df, label_df, test_size=test_size, random_state=random_state
            )

    Song_train, Song_eval, label_train, label_eval = train_test_split(
            Song_train, label_train, test_size=evaluation_size,
            random_state=random_state
            )

    Song_train.reset_index(drop=True, inplace=True)
    Song_eval.reset_index(drop=True, inplace=True)
    Song_test.reset_index(drop=True, inplace=True)
    label_train.reset_index(drop=True, inplace=True)
    label_eval.reset_index(drop=True, inplace=True)
    label_test.reset_index(drop=True, inplace=True)

    X_train, X_eval, X_test = [], [], []
    feature_labels = []
    for i, embed_type in enumerate(embed_types):
        embed_base = embed_type.split('_')[0]
        feature_labels = feature_labels + [f'{embed_base}_{label}' for label in
                               FEATURE_LABELS]
        train_embeds = get_embeddings(Song_train[embed_type])
        eval_embeds = get_embeddings(Song_eval[embed_type])
        test_embeds = get_embeddings(Song_test[embed_type])

        if i == 0:
            X_train = train_embeds
            X_eval = eval_embeds
            X_test = test_embeds
        else:
            X_train = [np.concatenate([X_train[j], train_embeds[j]]) for j in
                       range(len(X_train))]
            X_eval = [np.concatenate([X_eval[j], eval_embeds[j]]) for j in
                      range(len(X_eval))]
            X_test = [np.concatenate([X_test[j], test_embeds[j]]) for j in
                      range(len(X_test))]


    y_train = make_labels(label_train, Song_train[VCE])
    y_eval = make_labels(label_eval, Song_eval[VCE])
    y_test = make_labels(label_test, Song_test[VCE])
    return X_train, X_eval, X_test, y_train, y_eval, y_test, feature_labels

'''
[Helper function for run_xgb] Scales X input sets (normalizes) and labels y
label sets (turns strings into ints).
'''
def scale_label_sets(
        scaler=StandardScaler(),
        X_train=[],
        X_eval=[],
        X_test=[],
        label_encoder=LabelEncoder(),
        y_train=[],
        y_eval=[],
        y_test=[],
        ):
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    X_test_scaled = scaler.transform(X_test)
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_eval_encoded = label_encoder.transform(y_eval)
    y_test_encoded = label_encoder.transform(y_test)

    return (X_train_scaled, X_eval_scaled, X_test_scaled, y_train_encoded,
            y_eval_encoded, y_test_encoded)
'''
[Helper function for run_xgb] Runs 5 fold random search cross validation.
'''
def cross_validation_hyperparams(
        X_train,
        y_train,
        model,
        params=PARAM_DISTRIBUTIONS,
        n_iter=100,
        scoring='f1_macro',
        n_jobs=N_JOBS,
        verbose=2,
        random_state=RANDOM_STATE,
        ):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCVWithProgress(
            model,
            params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state
            )
    start_time = time.time()
    logging.info(f'Starting cross validation to find hyperparams.')
    random_search.fit(X_train, y_train)
    logging.info(f'RandomSearchCV took {time.time() - start_time:.2f} seconds.')
    logging.info(f"Best parameters: {random_search.best_params_}")
    logging.info(f"Best CV score: {random_search.best_score_:.4f}")

    return random_search

'''
[Helper function for run_xgb] Exports a confusion matrix, feature importance
graph, and decision tree for XGBoost performance metrics.
'''
def plot_findings(
        y_test_encoded,
        y_pred,
        target_names=None,
        model=None,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
        feature_importance_path=FEATURE_IMPORTANCE_PATH,
        tree_path=XGBOOST_TREE_PATH
        ):
    report = classification_report(y_test_encoded, y_pred,
                                   target_names=target_names)
    logging.info(f'\n{report}')
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    # plot confusion matrix
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix: Actual vs Predicted')
    plt.savefig(confusion_matrix_path)
    plt.close()
    # plot feature importance
    fig, ax = plt.subplots(figsize=(30,10))
    plot_importance(model, ax=ax, max_num_features=10, importance_type='weight')
    plt.title('Feature Importance')
    plt.savefig(feature_importance_path)
    plt.tight_layout()
    plt.close()
    #
    # plot a single tree
    fig, ax = plt.subplots(figsize=(50, 10))
    xgb.plot_tree(model, num_trees=0, ax=ax)  # num_trees=0 plots the first tree
    fig.savefig(tree_path, dpi=600, bbox_inches='tight')
    plt.close()

'''
Trains a model on the XGBoost framework.
'''
def run_xgb(
        X_train=None,
        X_eval=None,
        X_test=None,
        y_train=None,
        y_eval=None,
        y_test=None,
        target_names=None,
        feature_names=FEATURE_LABELS,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
        feature_importance_path=FEATURE_IMPORTANCE_PATH,
        tree_path=XGBOOST_TREE_PATH
        ):
    logging.info('Starting to run XGBoost')
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    X_train_scaled, X_eval_scaled, X_test_scaled, \
    y_train_encoded, y_eval_encoded, y_test_encoded = scale_label_sets(
             scaler=scaler,
             X_train=X_train,
             X_eval=X_eval,
             X_test=X_test,
             label_encoder=label_encoder,
             y_train=y_train,
             y_eval=y_eval,
             y_test=y_test
             )

    logging.info('Creating the model.')
    model = XGBClassifier(use_label_encoder=None, eval_metric='mlogloss',
                          objective='multi:softprob', n_jobs=N_JOBS)

    random_search = cross_validation_hyperparams(X_train, y_train_encoded,
                                                 model)
    best_params = random_search.best_params_

    model = XGBClassifier(**best_params,
                          use_label_encoder=None,
                          eval_metric='mlogloss',
                          objective='multi:softprob',
                          n_jobs=N_JOBS,
                          early_stopping_rounds=10)

    model.fit(X_train_scaled, y_train_encoded, eval_set=[(X_eval_scaled, y_eval_encoded)],
              verbose=True)
    model.get_booster().feature_names = feature_names

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    logging.info(f'Accuracy: {accuracy:.2f}')

    if target_names is None:
        target_names = [str(label) for label in label_encoder.classes_]

    plot_findings(y_test_encoded=y_test_encoded, y_pred=y_pred,
                  target_names=target_names, model=model,
                  confusion_matrix_path=confusion_matrix_path,
                  feature_importance_path=feature_importance_path,
                  tree_path=tree_path)

    return model

if __name__ == '__main__':
    # Force reconfiguration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    create_logger(log_location='logs/model_creation.log')

    start_time = time.time()
    file_path = 'machine_data/df_clipembed_recordingid_producermd.pkl'

    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    # adding more banned words
    pattern = '|'.join(ADDITIONAL_BANNED_WORDS)

    # Filter out rows where 'song_name' contains any banned word (case insensitive)
    df_filtered = df[~df['song_name'].str.contains(pattern, case=False, na=False)]

    # debug section; set GET_REMOVED_SONGS to True if needed
    if GET_REMOVED_SONGS:
        filtered_out_rows = df[df['song_name'].str.contains(pattern, case=False, na=False)]
        print("\nNames filtered out:")
        print(filtered_out_rows['song_name'].tolist())

    df = df_filtered
    generations = df['generation']
    train_eval_test_split = create_train_eval_test_split(
        df,
        generations,
        embed_types=[VCE, ICE],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        )

    X_train, X_eval, X_test, y_train, y_eval, y_test, feature_labels = train_eval_test_split

    model = run_xgb(
            X_train=X_train,
            X_eval=X_eval,
            X_test=X_test,
            y_train=y_train,
            y_eval=y_eval,
            y_test=y_test,
            feature_names=feature_labels,
            confusion_matrix_path='figures/generation_confusion_matrix.png',
            feature_importance_path='figures/generation_feature_importance.png'
            )
    logging.info(f'It took {(time.time() - start_time) / 60} minutes to run XGBoost.')
