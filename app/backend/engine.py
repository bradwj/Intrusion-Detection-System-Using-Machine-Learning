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


class Model:
    parameter_values = {}

    def __init__(self):
        if not hasattr(self, "name"):
            raise NotImplementedError("Model must have a 'name' attribute")
        if not hasattr(self, "parameters"):
            self.parameters = []
        # set parameter values to default
        for param in self.parameters:
            self.parameter_values[param.name] = param.default

    @classmethod
    def get_models(self):
        return [model.to_dict() for model in self.__subclasses__()]

    @classmethod
    def get_parameters(self):
        return [param.to_dict() for param in self.parameters]

    @classmethod
    def to_dict(self):
        return {
            "name": self.name,
            "parameters": self.get_parameters(),
        }

    @classmethod
    def get_model(self, model_name):
        model = None
        for model_class in Model.__subclasses__():
            if model_class.name == model_name:
                model = model_class()
                break
        else:
            return (
                None,
                f"'{model_name}' is not a valid model. Available models: {[model.name for model in self.__subclasses__()]}",
            )

        return model, None

    def check_invalid_parameters(self, input_parameters):
        # input_parameters in form of:
        # ["param1": value, "param2": value, "param3": value]
        nonexistent_params = []
        invalid_params = []
        for key, value in input_parameters.items():
            if key not in [p.name for p in self.parameters]:
                nonexistent_params.append(key)
            else:
                # get parameter object from name
                param_obj = next((p for p in self.parameters if p.name == key), None)
                if not isinstance(value, param_obj.python_type):
                    invalid_params.append(key)

        if nonexistent_params:
            return f"Parameters {nonexistent_params} do not exist for model '{self.name}'. Valid parameters: {[p.name for p in self.parameters]}"
        if invalid_params:
            return f"Parameter values {invalid_params} are not valid for model '{self.name}'. Please provide parameters with their specified data types: {self.get_parameters()}"

        return None

    def set_parameters(self, input_parameters):
        if err := self.check_invalid_parameters(input_parameters):
            return err

        for param in self.parameters:
            if param.name in input_parameters:
                self.parameter_values[param.name] = input_parameters[param.name]


class ParameterType(Enum):
    int = "int"
    float = "float"
    str = "str"


class Parameter:
    # TODO: add valid range for values
    def __init__(self, name: str, dtype: ParameterType, default=None):
        self.name = name
        self.dtype = dtype
        self.default = default
        self.python_type = eval(dtype.name)

    def to_dict(self):
        return {
            "name": self.name,
            "dtype": self.dtype.name,
            "default": self.default,
        }


class LCCDE(Model):
    name = "LCCDE"
    parameters = [
        Parameter("num_leaves", ParameterType.int, 10),
        Parameter("min_data_in_leaf", ParameterType.int, 10),
        Parameter("max_depth", ParameterType.int, 10),
    ]

    # parameter_values is dict with {param.name: param_value} for each param in parameters
    def run(self):
        logger.info(
            f"Running {self.name} model with parameters: {self.parameter_values}"
        )
        # TODO: add code from ipynb to run the model

        return {
            "model": self.name,
            "parameters": [
                {"name": k, "value": v} for k, v in self.parameter_values.items()
            ],
            "results": {"accuracy": 0.9, "f1": 0.8},
        }, None


# TODO
class MTH(Model):
    name = "MTH"
    parameters = []


# TODO
class TreeBased(Model):
    name = "TreeBased"
    parameters = []


# previous implementation for reference (TODO: remove after refactoring):

# class MLEngine:

#     def __init__(self):
#         logger.info("*** Initializing ML Engine ***")
#         self.load_dataset()

#     def load_dataset(self):
#         logger.info("Loading dataset into memory")
#         self.dataset_df = pd.read_csv("./data/CICIDS2017_sample_km.csv")
#         X = self.dataset_df.drop(["Label"], axis=1)
#         y = self.dataset_df["Label"]
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, train_size=0.8, test_size=0.2, random_state=0
#         )  # shuffle=False

#         logger.info(
#             f"y_train value counts before SMOTE resampling:\n{pd.Series(self.y_train).value_counts()}"
#         )
#         smote = SMOTE(n_jobs=-1, sampling_strategy={2: 1000, 4: 1000})
#         self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

#         logger.info(
#             f"y_train value counts after SMOTE resampling:\n{pd.Series(self.y_train).value_counts()}"
#         )

#     def validate_parameters(self, engine_model, input_parameters):
#         valid_params = [param["name"] for param in self.PARAMETERS[engine_model]]
#         invalid_param_names = []
#         invalid_param_values = {}
#         for key, value in input_parameters:
#             if key not in valid_params:
#                 invalid_param_names.push(key)
#             try:
#                 dtype = self.PARAMETERS[engine_model][key]
#                 parsed_value = None
#                 if ParameterType(dtype):
#                     match ParameterType(dtype):
#                         case ParameterType.int:
#                             parsed_value = int(value)
#                         case ParameterType.float:
#                             parsed_value = float(value)
#                         case ParameterType.str:
#                             parsed_value = str(value)
#                 if not parsed_value:
#                     invalid_param_values[key] = value

#         if invalid_param_names or invalid_param_values:
#             err_msg = ""
#             if invalid_param_names:
#                 err_msg += f"Parameters names {invalid_param_names} are not a valid. Available parameters: {valid_params}."
#             if invalid_param_values:
#                 err_msg += f"\nParameter value data types are not valid: {invalid_param_values}. Please provide parameters with their specified data types: {self.PARAMETERS[engine_model]}"

#     def validate_model_name(self, model_name):
#         if model_name not in EngineModel._member_names_:
#             return f"'{model_name}' is not a valid model. Available models: {EngineModel._member_names_}",


#     def run_engine(self, model_name, input_parameters):
#         if err_msg := self.validate_model_name(model_name):
#             return {"error": err_msg}

#         self.engine_model = EngineModel(model_name)
#         model = None
#         if self.engine_model == EngineModel.lightgbm:
#             model = lgb.LGBMClassifier()
#         elif self.engine_model == EngineModel.xgboost:
#             model = xgb.XGBClassifier()
#         elif self.engine_model == EngineModel.catboost:
#             model = cbt.CatBoostClassifier()
#         else:
#             return None

#         if not self.validate_parameters(self.engine_model, )

#         start_time = dt.datetime.now()
#         total_start_time = train_start_time = time.time()
#         model.fit(self.X_train, self.y_train)
#         train_duration = time.time() - train_start_time

#         test_start_time = time.time()
#         self.y_pred = model.predict(self.X_test)
#         test_duration = time.time() - test_start_time

#         print(classification_report(self.y_test, self.y_pred))
#         print("Accuracy: " + str(accuracy_score(self.y_test, self.y_pred)))
#         print(
#             "Precision: "
#             + str(precision_score(self.y_test, self.y_pred, average="weighted"))
#         )
#         print(
#             "Recall: " + str(recall_score(self.y_test, self.y_pred, average="weighted"))
#         )
#         print(
#             "Average F1: " + str(f1_score(self.y_test, self.y_pred, average="weighted"))
#         )
#         print(
#             "F1 for each type of attack: "
#             + str(f1_score(self.y_test, self.y_pred, average=None))
#         )
#         cb_f1 = f1_score(self.y_test, self.y_pred, average=None)

#         # TODO: figure out how to return the confusion matrix (if wanted)
#         # Plot the confusion matrix
#         cm = confusion_matrix(self.y_test, self.y_pred)
#         f, ax = plt.subplots(figsize=(5, 5))
#         sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
#         plt.xlabel("y_pred")
#         plt.ylabel("y_true")
#         # plt.show()
#         total_duration = time.time() - total_start_time
#         end_time = dt.datetime.now()

#         return {
#             "model": model_name,
#             "start_time": str(start_time),
#             "end_time": str(end_time),
#             "train_duration": train_duration,
#             "test_duration": test_duration,
#             "total_duration": total_duration,
#             "results": {
#                 "accuracy": accuracy_score(self.y_test, self.y_pred),
#                 "precision": precision_score(
#                     self.y_test, self.y_pred, average="weighted"
#                 ),
#                 "recall": recall_score(self.y_test, self.y_pred, average="weighted"),
#                 "avg_f1": f1_score(self.y_test, self.y_pred, average="weighted"),
#                 "categorical_f1": f1_score(
#                     self.y_test, self.y_pred, average=None
#                 ).tolist(),
#             },
#         }
