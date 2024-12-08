# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python [conda env:522]
#     language: python
#     name: conda-env-522-py
# ---

# +
import click
# import os
# import altair as alt
# import altair_ally as aly
import numpy as np
import pandas as pd
# import pickle
# from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
# from deepchecks.tabular import Dataset

# from ucimlrepo import fetch_ucirepo 
# from sklearn.model_selection import train_test_split

from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

from sklearn.metrics import accuracy_score
# from joblib import dump

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# import pandera as pa

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--results-to', type=str, help="Path to directory where the model's best parameter and accuracy score will be written to")
# @click.option('--preprocessor_to', type=str, help="Path to preprocessor object")
# @click.option('--columns-to-drop', type=str, help="Optional: columns to drop")
# @click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
# @click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
# @click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=522)
def model_and_result(training_data, test_data, results_to, seed):
    '''Fits a wine quality logistic regression model to the training data 
    and evaluates the model on the test data with accuracy score.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read in training and test data
    train_df = pd.read_csv(training_data)
    test_df = pd.read_csv(test_data)
    # cancer_preprocessor = pickle.load(open(preprocessor, "rb"))


    # Split dataset
    X_train, X_test, y_train, y_test = (train_df.drop(columns='quality'), test_df.drop(columns='quality'),
                                        train_df['quality'], test_df['quality']
                                        )
    
    numeric_features = X_train.select_dtypes(include='number').columns.tolist()
    binary_features = ['color']
    
    # Make column transformer
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop='if_binary'), binary_features),
        (StandardScaler(), numeric_features)
    )
    # pickle.dump(preprocessor, open(os.path.join(preprocessor_to, "wine_quality_preprocessor.pickle"), "wb"))
    
    # Make pipeline using StandardScaler and LogisticRegression
    model = make_pipeline(
        preprocessor,
        LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
    )

    # Define parameter distribution
    param_dist = {
        'logisticregression__C': stats.uniform(0.001, 100),
    }
    
    # Perform randomized search
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       cv=3, # The least populated class in y has only 4 members, which is less than n_splits=5.
                                       n_iter=50,
                                       scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    
    # Best parameters
    print("Best Parameters:", random_search.best_params_)

    # with open(os.path.join(pipeline_to, "wine_quality_pipeline.pickle"), 'wb') as f:
    #     pickle.dump(random_search, f)
    
    # scaled_train = preprocessor.transform(X_train)
    # scaled_test = preprocessor.transform(X_test)
    # scaled_train.to_csv(os.path.join(data_to, "scaled_wine_quality_train.csv"), index=False)
    # scaled_test.to_csv(os.path.join(data_to, "scaled_wine_quality_test.csv"), index=False)

    # Evaluate
    y_pred = random_search.predict(X_test)
    accuracy_score(y_test, y_pred)

    # Save best parameter and accuracy scores
    model_results = pd.DataFrame({'best_C': [random_search.best_params_], {'accuracy': [accuracy_score]})
    model_results.to_csv(os.path.join(results_to, "model_results.csv"), index=False)

if __name__ == '__main__':
    model_and_result()
# -


