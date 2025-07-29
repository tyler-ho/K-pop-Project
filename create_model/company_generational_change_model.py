"""
This file zooms in the work done in model_creation.py onto specific companies
under all-generation classification and two-generation classification.

Main functions:
    generation_classification
"""
import logging
import pickle
import os
import time

# IMPORTANT: Configure environment before scikit-learn import
# These settings affect joblib's multiprocessing behavior
os.environ['JOBLIB_VERBOSITY'] = '100'   # Enable verbose joblib output
os.environ['LOKY_PICKLER'] = 'pickle'    # Use pickle for serialization
os.environ['PYTHONUNBUFFERED'] = '1'     # Disable output buffering

import numpy as np
import pandas as pd

from classes.constants import (
        BANNED_WORDS,
        ICE,
        VCE,
)
from classes.logger import create_logger
from create_model.model_creation import (
        create_train_eval_test_split,
        run_xgb,
)

def no_fifth_generation():
    """[Helper function for generation_classification]
    This function creates a lambda function that reclassifies all 5th
    generation as 4th generation.
    """
    return lambda generation: 4 if generation > 4 else generation

def two_generations(focus_gen):
    """ [Helper function for generation_classification]
    This function creates a lambda function to create a two-way classification.
    """
    return lambda gen: 0 if gen != focus_gen else 1

def generation_classification(company_list, generations, classification_type,
                              embed_types=[VCE]):
    """
    Given a specific list of Songs from a company, a list of generations, and a
    type of classification, runs XGBoost.

    Example usage: Run all generation classification on JYP. Run two-way
    classification on 1st generation vs all other generations within SM.
    """
    train_eval_test_split = create_train_eval_test_split(
        company_list,
        generations,
        embed_types=embed_types,
    )

    X_train, X_eval, X_test, \
            y_train, y_eval, y_test, feature_names = train_eval_test_split
    CM_path = f'figures/{classification_type}_classification_confusion_matrix.png'
    FI_path = f'figures/{classification_type}_classification_feature_importance.png'
    tree_path = f'figures/{classification_type}_classification_tree.png'

    model = run_xgb(
        X_train=X_train,
        X_eval=X_eval,
        X_test=X_test,
        y_train=y_train,
        y_eval=y_eval,
        y_test=y_test,
        feature_names=feature_names,
        confusion_matrix_path=CM_path,
        feature_importance_path=FI_path,
        tree_path=tree_path
    )
    return model

if __name__ == '__main__':
    create_logger('logs/company_generational_change_model.log')
    file_path = 'machine_data/df_clipembed_recordingid_producermd.pkl'

    with open(file_path, 'rb') as f:
        df = pickle.load(f)

    # adding more banned words
    pattern = '|'.join(BANNED_WORDS)

    # Filter out rows where 'song_name' contains any banned word (case insensitive)
    df_filtered = df[~df['song_name'].str.contains(pattern, case=False, na=False)]

    df = df_filtered
    companies = ['JYP', 'YG', 'SM']
    embed_types = [VCE, ICE]
    for company in companies:
        logging.info(f'Starting on company: {company}.')
        four_way_start_time = time.time()
        company_list = df[df['company'] == company]
        no_fifth_gen = no_fifth_generation()
        generations = company_list['generation'].apply(no_fifth_gen)
        all_gen_class_type = company + '_all_generation'

        if not os.path.exists(f'models/{all_gen_class_type}_xgb_model.xgb'):
            model = generation_classification(company_list,
                                              generations,
                                              all_gen_class_type,
                                              embed_types=embed_types)
            model.save_model(f'models/{all_gen_class_type}_xgb_model.xgb')

        four_way_end_time = time.time()
        four_way_duration = four_way_end_time - four_way_start_time
        logging.info(
                f'It took {four_way_duration / 60.0:.2f} minutes to run all '
                f'generation classification on {company}.'
        )
        for focus_generation in np.sort(generations.unique()):
            two_way_start_time = time.time()
            two_gens = two_generations(focus_generation)
            two_classes = generations.apply(two_gens)
            two_gen_class_type = f'{company}_two_generation_focus_gen_{focus_generation}'
            if not os.path.exists(f'models/{two_gen_class_type}_xgb_model.xgb'):
                model = generation_classification(company_list,
                                                  two_classes,
                                                  two_gen_class_type,
                                                  embed_types=embed_types)
                model.save(f'models/{two_gen_class_type}_xgb_model.xgb')

            two_way_end_time = time.time()
            two_way_duration = two_way_end_time - two_way_start_time
            logging.info(
                    f'It took {(two_way_duration / 60.0):.2f} minutes to run two way '
                    f'classification on {focus_generation} for {company}.'
            )
