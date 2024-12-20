import pandas as pd
import pytest, sklearn
import scipy.stats as stats
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.find_best_model import find_best_model

# create sample training data
train_df = pd.read_csv('data/processed/training_set.csv').drop(columns='color')[:10]
X_train, y_train = (train_df.drop(columns='quality'), train_df['quality'])

def sample_pipeline():
    """Creates a sample sklearn pipeline with Logistic Regression for testing
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Sample pipeline containing Logistic Regression
    """
    return make_pipeline(
        LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
    )

def sample_best_C(model, range, cv, n_iter, scoring_metric, seed):
    """Returns the best C found through randomized search given parameters in test cases
    
    Returns
    -------
    float
        C value of tuned Logistic Regression inside given pipeline
    """
    search_result = RandomizedSearchCV(model, param_distributions={'logisticregression__C': range},
                                       cv=cv,
                                       n_iter=n_iter,
                                       scoring=scoring_metric,
                                       random_state=seed)
    search_result.fit(X_train, y_train)
    return search_result.best_params_['logisticregression__C']


def test_X_train_incorrect_type():
    """Raises error when X_train is not a pandas data frame."""
    with pytest.raises(TypeError):
        find_best_model([], y_train, sample_pipeline(), stats.uniform(0.001, 100), 3, 50, 'accuracy', 42)

def test_y_train_incorrect_type():
    """Raises error when y_train is not a pandas series."""
    with pytest.raises(TypeError):
        find_best_model(X_train, [], sample_pipeline(), stats.uniform(0.001, 100), 3, 50, 'accuracy', 42)

def test_model_incorrect_type():
    """Raises error when model is not a sklearn pipeline."""
    with pytest.raises(TypeError):
        find_best_model(X_train, y_train, [], stats.uniform(0.001, 100), 3, 50, 'accuracy', 42)

def test_range_incorrect_type():
    """Raises error when range is not a numpy array or scipy.stats.uniform object."""
    with pytest.raises(TypeError):
        find_best_model(X_train, y_train, sample_pipeline(), [], 3, 50, 'accuracy', 42)

def test_cv_incorrect_type():
    """Raises error when cv is not an integer."""
    with pytest.raises(TypeError):
        find_best_model(X_train, y_train, sample_pipeline(), stats.uniform(0.001, 100), [], 50, 'accuracy', 42)
    
def test_n_iter_incorrect_type():
    """Raises error when n_iter is not an integer."""
    with pytest.raises(TypeError):
        find_best_model(X_train, y_train, sample_pipeline(), stats.uniform(0.001, 100), 3, [], 'accuracy', 42)

def test_scoring_metric_incorrect_type():
    """Raises error when scoring_metric is not a string."""
    with pytest.raises(TypeError):
        find_best_model(X_train, y_train, sample_pipeline(), stats.uniform(0.001, 100), 3, 50, [], 42)

def test_seed_incorrect_type():
    """Raises error when seed is not an integer."""
    with pytest.raises(TypeError):
        find_best_model(X_train, y_train, sample_pipeline(), stats.uniform(0.001, 100), 3, 50, 'accuracy', [])

def test_find_best_model_success():
    """Tests successfully tuned pipeline has approximately the same C produced through RandomizedSearchCV using the same random seed."""
    tuned_model = find_best_model(X_train, y_train, sample_pipeline(), stats.uniform(0.001, 100), 3, 50, 'accuracy', 42)
    assert isinstance(tuned_model, sklearn.model_selection.RandomizedSearchCV)
    assert tuned_model.best_params_['logisticregression__C'] == sample_best_C(sample_pipeline(), stats.uniform(0.001, 100), 3, 50, 'accuracy', 42)