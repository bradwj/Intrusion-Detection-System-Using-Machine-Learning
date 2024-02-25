import time
import datetime as dt
import logging
from enum import Enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
import time
from river import stream
from statistics import mode
from imblearn.over_sampling import SMOTE


logger = logging.getLogger("server.engine")


class EngineModels(Enum):
    lightgbm = "lightgbm"
    catboost = "catboost"
    xgboost = "xgboost"


class MLEngine:

    def __init__(self):
        logger.info("*** Initializing ML Engine ***")
        self.load_dataset()

    def load_dataset(self):
        logger.info("Loading dataset into memory")
        self.dataset_df = pd.read_csv("./data/CICIDS2017_sample_km.csv")
        X = self.dataset_df.drop(["Label"], axis=1)
        y = self.dataset_df["Label"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=0.8, test_size=0.2, random_state=0
        )  # shuffle=False

        logger.info(
            f"y_train value counts before SMOTE resampling:\n{pd.Series(self.y_train).value_counts()}"
        )
        smote = SMOTE(n_jobs=-1, sampling_strategy={2: 1000, 4: 1000})
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        logger.info(
            f"y_train value counts after SMOTE resampling:\n{pd.Series(self.y_train).value_counts()}"
        )

    def run_engine(self, model_name):
        engine_model = EngineModels(model_name)
        model = None
        if engine_model == EngineModels.lightgbm:
            model = lgb.LGBMClassifier()
        elif engine_model == EngineModels.xgboost:
            model = xgb.XGBClassifier()
        elif engine_model == EngineModels.catboost:
            model = cbt.CatBoostClassifier()
        else:
            return None

        start_time = dt.datetime.now()
        total_start_time = train_start_time = time.time()
        model.fit(self.X_train, self.y_train)
        train_duration = time.time() - train_start_time

        test_start_time = time.time()
        self.y_pred = model.predict(self.X_test)
        test_duration = time.time() - test_start_time

        print(classification_report(self.y_test, self.y_pred))
        print("Accuracy: " + str(accuracy_score(self.y_test, self.y_pred)))
        print(
            "Precision: "
            + str(precision_score(self.y_test, self.y_pred, average="weighted"))
        )
        print(
            "Recall: " + str(recall_score(self.y_test, self.y_pred, average="weighted"))
        )
        print(
            "Average F1: " + str(f1_score(self.y_test, self.y_pred, average="weighted"))
        )
        print(
            "F1 for each type of attack: "
            + str(f1_score(self.y_test, self.y_pred, average=None))
        )
        cb_f1 = f1_score(self.y_test, self.y_pred, average=None)

        # TODO: figure out how to return the confusion matrix (if wanted)
        # Plot the confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()
        total_duration = time.time() - total_start_time
        end_time = dt.datetime.now()

        return {
            "model": model_name,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "train_duration": train_duration,
            "test_duration": test_duration,
            "total_duration": total_duration,
            "metrics": {
                "accuracy": accuracy_score(self.y_test, self.y_pred),
                "precision": precision_score(
                    self.y_test, self.y_pred, average="weighted"
                ),
                "recall": recall_score(self.y_test, self.y_pred, average="weighted"),
                "avg_f1": f1_score(self.y_test, self.y_pred, average="weighted"),
                # "f1": f1_score(self.y_test, self.y_pred, average=None),
            },
        }
