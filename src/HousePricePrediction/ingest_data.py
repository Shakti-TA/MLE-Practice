import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit


def fetch_housing_data(housing_url, housing_path):
    """
    This function creates the target directory (if it doesn't exist),
    downloads the housing dataset from the specified URL as a .tgz archive,
    and extracts its contents into the specified local path.

    Parameters:
    ----------
    housing_url : str
        The URL to download the housing dataset from.
    housing_path : str
        The local directory where the dataset will be saved and extracted.

    Returns:
    -------
    None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """
    This function read the housing dataset from the the target directory.

    Parameters:
    ----------
    housing_path : str
        The local directory where the dataset will be saved and extracted.

    Returns:
    -------
    pandas dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def str_split(housing, test_sizes=0.2):
    """
    This function strtified split the data on Median household income

    Parameters:
    ---------
    housing: pandas dataframe
    test_size: train-test split ratio

    Returns:
    Strat_train_set: Stratified Train set
    Strat_test_set: Stratified Test set
    """

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_sizes, random_state=42
        )
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    return strat_train_set, strat_test_set


class DataIngestion:
    def __init__(self):
        self.DOWNLOAD_ROOT = (
            "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        )
        self.HOUSING_PATH = os.path.join("datasets", "housing")
        self.HOUSING_URL = self.DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def initiate_data_ingestion(self):
        fetch_housing_data(self.HOUSING_URL, self.HOUSING_PATH)

        housing = load_housing_data(self.HOUSING_PATH)

        strat_train_set, strat_test_set = str_split(housing, 0.2)

        train_path = os.path.join("datasets/housing", "train.csv")
        test_path = os.path.join("datasets/housing", "test.csv")

        strat_train_set.to_csv(train_path, index=False, header=True)
        strat_test_set.to_csv(test_path, index=False, header=True)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
