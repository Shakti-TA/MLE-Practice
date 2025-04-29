import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeRegressor

from Script.logger import setup_logger

def get_features(housing):
    """
    Add new ratio-based features to the housing dataset.

    This function creates and appends three new features to the housing dataset:
    - `rooms_per_household`
    - `bedrooms_per_room`
    - `population_per_household`

    These features are derived from existing columns and may help improve model performance.

    Parameters
    ----------
    housing : pandas.DataFrame
        Training dataset after stratified train-test split.

    Returns
    -------
    housing : pandas.DataFrame
        DataFrame with three additional engineered features.
    """
    try:
        housing["rooms_per_household"] = (
            housing["total_rooms"] / housing["households"]
        )
        housing["bedrooms_per_room"] = (
            housing["total_bedrooms"] / housing["total_rooms"]
        )
        housing["population_per_household"] = (
            housing["population"] / housing["households"]
        )
        logger.info("New features: rooms_per_household, bedrooms_per_room and population_per_household added")
        return housing

    except Exception as e:
        logger.error("Unable to create new features!")


def SimpleImputing(housing):
    """
    Impute missing values in numerical features using the median.

    This function selects numerical columns from the housing dataset and imputes
    missing values in those columns using the median strategy.

    Parameters
    ----------
    housing : pandas.DataFrame
        Training dataset after feature engineering.

    Returns
    -------
    housing_num : pandas.DataFrame
        DataFrame containing only the numerical features.
    X : pandas.DataFrame
        DataFrame with missing values imputed using the median.
    """
    try:
        imputer = SimpleImputer(strategy="median")

        housing_num = housing.drop("ocean_proximity", axis=1)

        imputer.fit(housing_num)
        X = imputer.transform(housing_num)

        logger.info("Nmerical data separated and  Imputing is done")
        return housing_num, X

    except Exception as e:
        logger.error("Unable to do Imputation!")


def Linear_Regression(housing_prepared, housing_labels):
    """
    Train a Linear Regression model on the imputed training data.

    This function fits a Linear Regression model using the preprocessed training
    data and corresponding target labels.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame
        Imputed training data with engineered features.
    housing_labels : pandas.Series
        Target values corresponding to the training data.

    Returns
    -------
    None
    """
    try:
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        housing_predictions = lin_reg.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)

        lin_mae = mean_absolute_error(housing_labels, housing_predictions)

        print(f"Model: Lin_Reg with RMSE: {lin_rmse:.2f} and MAE: {lin_mae:.2f}")

        logger.info("Linear regression model is successfully implemented")
    except Exception as e:
        logger.error("Linear regression failed!")


def Tree_Regression(housing_prepared, housing_labels):
    """
    Train a Decision Tree Regressor on the imputed training data.

    This function fits a Decision Tree Regression model using the preprocessed
    training data and corresponding target labels.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame
        Imputed training dataset with engineered features.
    housing_labels : pandas.Series
        Target values corresponding to the training data.

    Returns
    -------
    None
    """
    try:
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)

        housing_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)
        print(f"Model: Tree_Regression with RMSE: {tree_rmse:.2f}")

        logger.info("Decision Tree model is successfully implemented")

    except Exception as e:
        logger.error("Decision Tree is failed!")


def Random_Forest(housing_prepared, housing_labels):
    """
    Train a Random Forest Regressor on the imputed training data.

    This function fits a Random Forest Regression model using the preprocessed
    training data and corresponding target labels.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame
        Imputed training dataset with engineered features.
    housing_labels : pandas.Series
        Target values corresponding to the training data.

    Returns
    -------
    None
    """

    try:
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)
        cvres = rnd_search.cv_results_
        print("Model: Random_Forest")
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        logger.info("Random Forest model is successfully implemented")

    except Exception as e:
        logger.error("Random Forest is failed!")


def Best_Model(housing_prepared, housing_labels):
    """
    Perform Grid Search to tune hyperparameters for Random Forest on the imputed training data.

    This function uses Grid Search to find the best hyperparameters for a Random Forest
    Regressor, based on the preprocessed training data and corresponding target labels.

    Parameters
    ----------
    housing_prepared : pandas.DataFrame
        Imputed training dataset with engineered features.
    housing_labels : pandas.Series
        Target values corresponding to the training data.

    Returns
    -------
    final_model : sklearn.ensemble.RandomForestRegressor
        The best Random Forest model found by Grid Search, with optimized hyperparameters.
    """

    try:
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {"bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4]},
        ]

        forest_reg = RandomForestRegressor(random_state=42)
        # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        grid_search.fit(housing_prepared, housing_labels)

        grid_search.best_params_
        cvres = grid_search.cv_results_

        print("Model: Grid_Search with Random_Forest")
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

        final_model = grid_search.best_estimator_

        return final_model

        logger.info("Best model is successfully implemented")

    except Exception as e:
        logger.error("Best model is failed!")


def saving(final_model, model_path):
    """
    Save the best model parameters to the artifacts.

    This function stores the best Random Forest model (with optimized hyperparameters)
    to the artifacts directory for later use or deployment.

    Parameters
    ----------
    final_model : sklearn.ensemble.RandomForestRegressor
        The best Random Forest model with optimized hyperparameters.

    Returns
    -------
    None
    """

    try:
        # Ensure 'artifacts' folder exists
        os.makedirs("artifacts", exist_ok=True)

        # Save the model
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)

        print(f"Model saved to: {model_path}")

        logger.info("Model saved as Pickel successfully")

    except Exception as e:
        logger.error("Failed to save best model!")

class Data_Train:
    def __init__(self, train_path, model_path):
        self.train_path = train_path
        self.housing_df = pd.read_csv(self.train_path)
        self.model_path = model_path

    def get_data_train(self):
        housing = self.housing_df.drop(
            "median_house_value", axis=1
        )  # drop labels for training set
        housing_labels = self.housing_df["median_house_value"].copy()

        # NUMERICAL COLUMNS
        housing_num, X = SimpleImputing(housing)
        housing_tr = pd.DataFrame(
            X,
            columns=housing_num.columns,
            index=housing.index)
        housing_tr = get_features(housing_tr)

        # CATEGORICAL COLUMNS
        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(
            pd.get_dummies(housing_cat, drop_first=True)
            )

        # MODEL TRAINING
        Linear_Regression(housing_prepared, housing_labels)
        Tree_Regression(housing_prepared, housing_labels)
        Random_Forest(housing_prepared, housing_labels)
        best_model_rnd_frst = Best_Model(housing_prepared, housing_labels)

        # SAVE THE BEST MODEL
        saving(best_model_rnd_frst, self.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate models performance on train data."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default = "datasets/housing/train.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default = "artifacts/final_model.pkl",
        help="Path to pickled model file"
    )

    parser.add_argument(
        "--log_to_file",
        type = bool,
        default = False,
    )

    parser.add_argument(
        "--log_to_console",
        type = bool,
        default = True,
    )

    parser.add_argument(
        "--log_level",
        type = str,
        default = 'DEBUG',
    )
    args = parser.parse_args()

    #logging
    logger = setup_logger(log_to_file=args.log_to_file,
                          log_to_console=args.log_to_console,
                          log_level=args.log_level)

    obj = Data_Train(args.train_data_path, args.model_path)
    obj.get_data_train()
