import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


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


class Score:
    def __init__(self, test_path, model_path):
        self.test_path = test_path
        self.model_path = model_path
        self.test_df = pd.read_csv(self.test_path)

    def get_data(self):
        housing = self.test_df.drop("median_house_value", axis=1)
        housing_labels = self.test_df["median_house_value"].copy()

        housing_num, X = SimpleImputing(housing)
        housing_tr = pd.DataFrame(X,
                                  columns=housing_num.columns,
                                  index=housing.index)
        housing_tr = get_features(housing_tr)

        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(
            pd.get_dummies(housing_cat, drop_first=True)
            )

        return housing_prepared, housing_labels

    def get_model_score(self, data, labels):
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        final_predictions = model.predict(data)
        final_mse = mean_squared_error(labels, final_predictions)
        final_rmse = np.sqrt(final_mse)

        print(f"Test data RMSE score: {final_rmse:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model performance on test data."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True, help="Path to pickled model file"
    )

    args = parser.parse_args()

    obj = Score(args.test_data_path, args.model_path)
    data, labels = obj.get_data()
    obj.get_model_score(data, labels)
