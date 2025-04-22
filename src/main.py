import os


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
    score_obj = Score(test_data_path)
    data, labels = score_obj.get_data()
    score_obj.get_model_score(data, labels)
