import os
import random
import unittest

import pandas as pd

from HousePricePrediction.ingest_data import load_housing_data, str_split
from HousePricePrediction.train import SimpleImputing, get_features
from HousePricePrediction.score import Score


class TestDataIngestion(unittest.TestCase):

    def setUp(self):
        # Set up the path to the test dataset
        self.housing_path = os.path.join("datasets", "housing")
        self.housing_data = load_housing_data(self.housing_path)

    def test_load_housing_data(self):
        # Check that data is loaded as a DataFrame
        self.assertIsInstance(self.housing_data, pd.DataFrame)

        # Check that it's not empty
        self.assertFalse(self.housing_data.empty)

        # Check specific expected columns
        expected_columns = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
            "ocean_proximity",
        ]
        for column in expected_columns:
            self.assertIn(column, self.housing_data.columns)

    def test_str_split(self):
        # Split the data
        train_set, test_set = str_split(self.housing_data, test_sizes=0.2)

        # Check types
        self.assertIsInstance(train_set, pd.DataFrame)
        self.assertIsInstance(test_set, pd.DataFrame)

        # Check combined size equals original size
        total_len = len(train_set) + len(test_set)
        self.assertEqual(total_len, len(self.housing_data))

        # Check that income_cat is in final sets
        self.assertIn("income_cat", train_set.columns)
        self.assertIn("income_cat", test_set.columns)


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.train_path = os.path.join("datasets/housing", "train.csv")
        self.train_df = pd.read_csv(self.train_path)

    def test_get_features(self):
        self.train_df = get_features(self.train_df)

        new_features = [
            "rooms_per_household",
            "bedrooms_per_room",
            "population_per_household",
        ]

        for feature in new_features:
            self.assertIn(feature, self.train_df.columns)

        # Pick a random row to validate formula logic
        rand_ind = random.randint(0, len(self.train_df) - 1)

        row = self.train_df.iloc[rand_ind]
        self.assertAlmostEqual(
            row["rooms_per_household"], row["total_rooms"] / row["households"]
        )
        self.assertAlmostEqual(
            row["bedrooms_per_room"],
            row["total_bedrooms"] / row["total_rooms"]
        )
        self.assertAlmostEqual(
            row["population_per_household"],
            row["population"] / row["households"]
        )

    def test_SimpleImputing(self):
        housing_num, imputed_df = SimpleImputing(self.train_df)

        num_cols = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
        ]

        cat_col = ["ocean_proximity"]

        for col in num_cols:
            self.assertIn(col, housing_num.columns)

        for cat in cat_col:
            self.assertNotIn(cat, housing_num.columns)


class TestScore(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join("datasets/housing", "test.csv")
        self.obj = Score(self.test_path)
        self.data, self.labels = self.obj.get_data()

    def test_get_model_score(self):
        final_rmse = self.obj.get_model_score(self.data, self.labels)

        # Check if final_rmse is equal to 48844.50
        self.assertEqual(int(final_rmse), 48844)


# if __name__ == '__main__':
#     unittest.main()
