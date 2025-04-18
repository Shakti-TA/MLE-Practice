import os
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error

from HousePricePrediction.ingest_data import DataIngestion
from HousePricePrediction.score import Score
from HousePricePrediction.train import Data_Train

if __name__ == "__main__":
    data_ingest_obj = DataIngestion()
    data_ingest_obj.initiate_data_ingestion()

    train_data_path = os.path.join("datasets/housing", "train.csv")
    data_train_obj = Data_Train(train_data_path)
    data_train_obj.get_data_train()

    test_data_path = os.path.join("datasets/housing", "test.csv")
    data_score_obj = Score(test_data_path)
    data, labels = data_score_obj.get_model_score()

    with open("artifacts/final_model.pkl", "rb") as f:
        model = pickle.load(f)

    final_predictions = model.predict(data)
    final_mse = mean_squared_error(labels, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print(f"Test data RMSE score: {final_rmse: .2f}")
