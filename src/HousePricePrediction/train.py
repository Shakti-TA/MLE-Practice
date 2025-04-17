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


def get_features(housing):
    """
    This function add new features: rooms_per_household,
                                    bedrooms_per_room,
                                    population_per_household,
    in the housing dataset.

    Parameters:
    ----------
    housing : pandas dataframe
              Train data after stratified train_test split
    Returns:
    -------
    housing : pandas dataframe
              dataframe with 3 new features
    """
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
        )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )
    return housing


def SimpleImputing(housing):
    """
    This function imputes the numerical feature columns with the median.

    Parameters:
    ----------
    housing : pandas dataframe
              Train data after feature engineering.
    Returns:
    -------
    housing_num : pandas dataframe
                dataframe with numerical features only
    X : pandas dataframe
       dataframe with imputed values
    """
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    return housing_num, X


def Linear_Regression(housing_prepared, housing_labels):
    """
    This function leverage Linear Regression on the Imputed train data.

    Parameters:
    ----------
    housing_prepared : pandas dataframe
                       Imputed Train data.
    housing_labels : pandas series
                     target_labels
    Returns:
    -------
    None
    """
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)

    print(f"Model: Lin_Reg with RMSE: {lin_rmse:.2f} and MAE: {lin_mae:.2f}")


def Tree_Regression(housing_prepared, housing_labels):
    """
    This function leverage Tree Regression on the Imputed train data.

    Parameters:
    ----------
    housing_prepared : pandas dataframe
                       Imputed Train data.
    housing_labels : pandas series
                     target_labels
    Returns:
    -------
    None
    """
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(f"Model: Tree_Regression with RMSE: {tree_rmse:.2f}")


def Random_Forest(housing_prepared, housing_labels):
    """
    This function leverage Random Forest on the Imputed train data.

    Parameters:
    ----------
    housing_prepared : pandas dataframe
                       Imputed Train data.
    housing_labels : pandas series
                     target_labels
    Returns:
    -------
    None
    """
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


def Best_Model(housing_prepared, housing_labels):
    """
    This function leverage Grid search with Random
    Forest on the Imputed train data.

    Parameters:
    ----------
    housing_prepared : pandas dataframe
                       Imputed Train data.
    housing_labels : pandas series
                     target_labels
    Returns:
    -------
    final_model: best model parameters
    """
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10],
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


def saving(final_model):
    """
    This function saves the best model parameters to the artifacts.

    Parameters:
    ----------
    final_model: Best Random_forest model parameters

    Returns:
    -------
    None
    """
    # Ensure 'artifacts' folder exists
    os.makedirs("artifacts", exist_ok=True)

    # Define file path directly in artifacts
    model_path = os.path.join("artifacts", "final_model.pkl")

    # Save the model
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    print(f"Model saved to: {model_path}")


class Data_Train:
    def __init__(self, train_path):
        self.train_path = train_path
        self.housing_df = pd.read_csv(self.train_path)

    def get_data_train(self):
        housing = self.housing_df.drop(
            "median_house_value", axis=1
        )  # drop labels for training set
        housing_labels = self.housing_df["median_house_value"].copy()

        # NUMERICAL COLUMNS
        housing_num, X = SimpleImputing(housing)
        housing_tr = pd.DataFrame(X,
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
        saving(best_model_rnd_frst)


if __name__ == "__main__":
    train_data_path = os.path.join("datasets/housing", "train.csv")
    print(train_data_path)
    obj = Data_Train(train_data_path)
    obj.get_data_train()
