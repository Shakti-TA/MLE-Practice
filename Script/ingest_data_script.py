import argparse
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

from Script.logger import setup_logger


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
    try:
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

        logger.info("Housing data fetched from the web and saved into the path: '%s'", housing_path)

    except Exception as e:

        logger.error("Unable to fetch data from the web: %s", str(e))


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
    try:
        csv_path = os.path.join(housing_path, "housing.csv")
        df = pd.read_csv(csv_path)
        logger.info("Housing data successfully read from '%s'", csv_path)
        return df
    except Exception as e:
        logger.error("Unable to read data or data is absent: %s", str(e))
        return None



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

    try:
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

        logger.info("Splitting is done!")

        return strat_train_set, strat_test_set
    except Exception as e:
        logger.error("Unable to perform train-test split: %s", str(e))
        return None, None


class DataIngestion:
    def __init__(self, path):
        self.DOWNLOAD_ROOT = (
            "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        )
        self.HOUSING_PATH = path
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
    parser = argparse.ArgumentParser(
        description="Download and split housing data with logging."
        )
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join("datasets", "housing"),
        help="Directory to store the housing data",
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

    obj = DataIngestion(args.path)
    obj.initiate_data_ingestion()
