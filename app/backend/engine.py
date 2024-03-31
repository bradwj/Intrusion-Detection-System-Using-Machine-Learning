import time
import datetime as dt
import logging
from enum import Enum

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
# )
# import lightgbm as lgb
# import catboost as cbt
# import xgboost as xgb
# import time
# from river import stream
# from statistics import mode
# from imblearn.over_sampling import SMOTE


logger = logging.getLogger("app.engine")
logger.setLevel(logging.INFO)


class Interval:
    def __init__(self, interval_string):
        # interval looks like "[min_val, max_val]" or "(min_val, max_val)" or "[min_val, max_val)"
        # '[' or ']' means inclusive, '(' or ')' means exclusive
        self.interval_string = interval_string
        try:
            lower_bound, upper_bound = interval_string[1:-1].split(",")
            self.lower_bound = float(lower_bound)
            self.upper_bound = float(upper_bound)
            if lower_bound > upper_bound:
                raise ValueError(
                    f"Invalid interval string: {interval_string}. Lower bound must be less than or equal to upper bound"
                )
            self.lower_bound_inclusive = interval_string[0] == "["
            self.upper_bound_inclusive = interval_string[-1] == "]"
        except ValueError:
            raise ValueError(f"Invalid interval string: {interval_string}")

    def __contains__(self, value):
        if self.lower_bound_inclusive and value < self.lower_bound:
            return False
        if not self.lower_bound_inclusive and value <= self.lower_bound:
            return False
        if self.upper_bound_inclusive and value > self.upper_bound:
            return False
        if not self.upper_bound_inclusive and value >= self.upper_bound:
            return False
        return True

    def __str__(self):
        return self.interval_string


LIGHTGBM_CLASSIFIER = "lightgbm_classifier"
XGBOOST_CLASSIFIER = "xgboost_classifier"
CATBOOST_CLASSIFIER = "catboost_classifier"

SUBMODEL_AVAILABLE_PARAMETERS = {
    # lightgbm parameters:
    # num_iterations, default = 100, type = int, constraints: num_iterations >= 0
    # learning_rate, default = 0.1, type = double, constraints: learning_rate > 0.0
    # num_leaves, default = 31, type = int, constraints: 1 < num_leaves <= 131072
    # max_depth, default = -1, type = int
    # min_data_in_leaf︎, default = 20, type = int, constraints: min_data_in_leaf >= 0
    LIGHTGBM_CLASSIFIER: {
        "num_iterations": {
            "dtype": "int",
            "default": 100,
            "range": "[0,inf)",
        },
        "learning_rate": {
            "dtype": "float",
            "default": 0.1,
            "range": "(0,inf)",
        },
        "num_leaves": {
            "dtype": "int",
            "default": 31,
            "range": "(1,131072]",
        },
        "max_depth": {"dtype": "int", "default": -1},
        "min_data_in_leaf": {
            "dtype": "int",
            "default": 20,
            "range": "[0,inf)",
        },
    },
    # XGBOOST parameters:
    # eta [default=0.3, alias: learning_rate, range: [0,1]]
    # gamma [default=0, alias: min_split_loss, range: [0,∞]]
    # max_depth [default=6, range: [0,∞]]
    # min_child_weight [default=1, range: [0,∞]]
    # subsample [default=1, range: (0,1]]
    # lambda [default=1, alias: reg_lambda, range: [0,∞]]
    # alpha [default=0, alias: reg_alpha, range: [0,∞]]
    # tree_method [default='auto', choices: {'auto', 'exact', 'approx', 'hist'}]
    # max_leaves [default=0, range: [0,∞]]
    XGBOOST_CLASSIFIER: {
        "eta": {"dtype": "float", "default": 0.3, "range": "[0,1]"},
        "gamma": {"dtype": "float", "default": 0, "range": "[0,inf)"},
        "max_depth": {"dtype": "int", "default": 6, "range": "[0,inf)"},
        "min_child_weight": {"dtype": "int", "default": 1, "range": "[0,inf)"},
        "subsample": {"dtype": "float", "default": 1, "range": "(0,1]"},
        "lambda": {"dtype": "float", "default": 1, "range": "[0,inf)"},
        "alpha": {"dtype": "float", "default": 0, "range": "[0,inf)"},
        "tree_method": {
            "dtype": "str",
            "default": "auto",
            "choices": ["auto", "exact", "approx", "hist"],
        },
        "max_leaves": {"dtype": "int", "default": 0, "range": "[0,inf)"},
    },
    CATBOOST_CLASSIFIER: {
        "boosting_type": {
            "dtype": "str",
            "default": "Plain",
        },
    },
}


class Model:
    def __init__(self):
        if not hasattr(self, "name"):
            raise NotImplementedError("Model must have a 'name' attribute")
        if not hasattr(self, "parameters"):
            self.parameters = {}

    @classmethod
    def get_models(self):
        return [model.to_dict() for model in self.__subclasses__()]

    @classmethod
    def get_parameter_metadata(self):
        param_metadata = {}
        for submodel, model_params in self.parameters.items():
            submodel_param_info = SUBMODEL_AVAILABLE_PARAMETERS[submodel]
            for param_name, param_value in model_params.items():
                submodel_param_info[param_name]["model_default"] = param_value
            param_metadata[submodel] = submodel_param_info

        return param_metadata

    @classmethod
    def to_dict(self):
        return {
            "name": self.name,
            "parameters": self.get_parameter_metadata(),
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

    def set_parameters(self, input_parameters):
        # validate and set parameters
        for submodel, submodel_params in input_parameters.items():
            if submodel not in self.parameters.keys():
                return f"The {self.name} model does not use submodel '{submodel}'. Valid submodels: {list(self.parameters.keys())}"

            for param_name, param_value in submodel_params.items():
                # validate parameter name
                if param_name not in SUBMODEL_AVAILABLE_PARAMETERS[submodel]:
                    return f"Parameter '{param_name}' does not exist for submodel '{submodel}'. Available parameters: {list(SUBMODEL_AVAILABLE_PARAMETERS[submodel].keys())}"

                # validate parameter value
                expected_type = SUBMODEL_AVAILABLE_PARAMETERS[submodel][param_name][
                    "dtype"
                ]
                if not isinstance(param_value, eval(expected_type)):
                    return f"Parameter value '{param_value}' for '{param_name}' is not of type {expected_type}"

                # validate parameter range
                if "range" in SUBMODEL_AVAILABLE_PARAMETERS[submodel][param_name]:
                    expected_range = SUBMODEL_AVAILABLE_PARAMETERS[submodel][
                        param_name
                    ]["range"]
                    if param_value not in Interval(expected_range):
                        return f"Parameter value '{param_value}' for '{param_name}' is not within the expected range {expected_range}"

                # validate parameter choices
                if "choices" in SUBMODEL_AVAILABLE_PARAMETERS[submodel][param_name]:
                    expected_choices = SUBMODEL_AVAILABLE_PARAMETERS[submodel][
                        param_name
                    ]["choices"]
                    if param_value not in expected_choices:
                        return f"Parameter value '{param_value}' for '{param_name}' is not one of the expected choices {expected_choices}"

                # set parameter value
                logger.info(f"Setting parameter '{param_name}' to '{param_value}'")
                self.parameters[submodel][param_name] = param_value


# class Parameter:
#     def __init__(
#         self,
#         name: str,
#         dtype: type,
#         default=None,
#         # constraints: callable = None,
#         value=None,
#     ):
#         self.name = name
#         self.dtype = dtype
#         self.default = default
#         # self.constraints = constraints if constraints is not None else lambda x: True
#         self.value = value if value is not None else default

#     def get_metadata(self):
#         return {
#             "name": self.name,
#             "dtype": str(self.dtype).split("'")[1],
#             "default": self.default,
#             # "constraints": self.constraints,
#         }

#     def to_dict(self):
#         return self.get_metadata() | {"value": self.value}

# LIGHTGBM_CLASSIFIER_PARAMETERS = [
#     Parameter("num_iterations", int, default=100),
#     Parameter("learning_rate", float, default=0.1),
#     Parameter("num_leaves", int, default=31),
#     Parameter("max_depth", int, default=-1),
#     Parameter("min_data_in_leaf", int, default=20),
# ]


class LCCDE(Model):
    name = "LCCDE"
    parameters = {
        LIGHTGBM_CLASSIFIER: {},
        XGBOOST_CLASSIFIER: {},
        CATBOOST_CLASSIFIER: {"boosting_type": "Plain"},
    }

    def run(self):
        logger.info(f"Running {self.name} model with parameters: {self.parameters}")
        start_time = dt.datetime.now()

        ###### Code from LCCDE_IDS_GlobeCom22.ipynb
        import warnings

        warnings.filterwarnings("ignore")

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

        df = pd.read_csv("./data/CICIDS2017_sample_km.csv")

        X = df.drop(["Label"], axis=1)
        y = df["Label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, test_size=0.2, random_state=0
        )  # shuffle=False

        from imblearn.over_sampling import SMOTE

        smote = SMOTE(n_jobs=-1, sampling_strategy={2: 1000, 4: 1000})

        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Train the LightGBM algorithm
        import lightgbm as lgb

        lg = lgb.LGBMClassifier(**self.parameters[LIGHTGBM_CLASSIFIER])
        lg.fit(X_train, y_train)
        y_pred = lg.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("Accuracy of LightGBM: " + str(accuracy_score(y_test, y_pred)))
        print(
            "Precision of LightGBM: "
            + str(precision_score(y_test, y_pred, average="weighted"))
        )
        print(
            "Recall of LightGBM: "
            + str(recall_score(y_test, y_pred, average="weighted"))
        )
        print(
            "Average F1 of LightGBM: "
            + str(f1_score(y_test, y_pred, average="weighted"))
        )
        print(
            "F1 of LightGBM for each type of attack: "
            + str(f1_score(y_test, y_pred, average=None))
        )
        lg_f1 = f1_score(y_test, y_pred, average=None)

        # Plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show() # TODO: figure out how to return the confusion matrix (if wanted)

        # Train the XGBoost algorithm
        import xgboost as xgb

        xg = xgb.XGBClassifier(**self.parameters[XGBOOST_CLASSIFIER])

        X_train_x = X_train.values
        X_test_x = X_test.values

        xg.fit(X_train_x, y_train)

        y_pred = xg.predict(X_test_x)
        print(classification_report(y_test, y_pred))
        print("Accuracy of XGBoost: " + str(accuracy_score(y_test, y_pred)))
        print(
            "Precision of XGBoost: "
            + str(precision_score(y_test, y_pred, average="weighted"))
        )
        print(
            "Recall of XGBoost: "
            + str(recall_score(y_test, y_pred, average="weighted"))
        )
        print(
            "Average F1 of XGBoost: "
            + str(f1_score(y_test, y_pred, average="weighted"))
        )
        print(
            "F1 of XGBoost for each type of attack: "
            + str(f1_score(y_test, y_pred, average=None))
        )
        xg_f1 = f1_score(y_test, y_pred, average=None)

        # Plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show() # TODO: figure out how to return the confusion matrix (if wanted)

        # Train the CatBoost algorithm
        import catboost as cbt

        # cb = cbt.CatBoostClassifier(verbose=0, boosting_type="Plain")
        # cb = cbt.CatBoostClassifier()
        cb = cbt.CatBoostClassifier(verbose=0, **self.parameters[CATBOOST_CLASSIFIER])

        cb.fit(X_train, y_train)
        y_pred = cb.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("Accuracy of CatBoost: " + str(accuracy_score(y_test, y_pred)))
        print(
            "Precision of CatBoost: "
            + str(precision_score(y_test, y_pred, average="weighted"))
        )
        print(
            "Recall of CatBoost: "
            + str(recall_score(y_test, y_pred, average="weighted"))
        )
        print(
            "Average F1 of CatBoost: "
            + str(f1_score(y_test, y_pred, average="weighted"))
        )
        print(
            "F1 of CatBoost for each type of attack: "
            + str(f1_score(y_test, y_pred, average=None))
        )
        cb_f1 = f1_score(y_test, y_pred, average=None)

        # Plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show() # TODO: figure out how to return the confusion matrix (if wanted)

        # Leading model list for each class
        model = []
        for i in range(len(lg_f1)):
            if max(lg_f1[i], xg_f1[i], cb_f1[i]) == lg_f1[i]:
                model.append(lg)
            elif max(lg_f1[i], xg_f1[i], cb_f1[i]) == xg_f1[i]:
                model.append(xg)
            else:
                model.append(cb)

        def do_LCCDE(X_test, y_test, m1, m2, m3):
            i = 0
            t = []
            m = []
            yt = []
            yp = []
            l = []
            pred_l = []
            pro_l = []

            # For each class (normal or a type of attack), find the leader model
            for xi, yi in stream.iter_pandas(X_test, y_test):

                xi2 = np.array(list(xi.values()))
                y_pred1 = m1.predict(
                    xi2.reshape(1, -1)
                )  # model 1 (LightGBM) makes a prediction on text sample xi
                y_pred1 = int(y_pred1[0])
                y_pred2 = m2.predict(
                    xi2.reshape(1, -1)
                )  # model 2 (XGBoost) makes a prediction on text sample xi
                y_pred2 = int(y_pred2[0])
                y_pred3 = m3.predict(
                    xi2.reshape(1, -1)
                )  # model 3 (Catboost) makes a prediction on text sample xi
                y_pred3 = int(y_pred3[0])

                p1 = m1.predict_proba(
                    xi2.reshape(1, -1)
                )  # The prediction probability (confidence) list of model 1
                p2 = m2.predict_proba(
                    xi2.reshape(1, -1)
                )  # The prediction probability (confidence) list of model 2
                p3 = m3.predict_proba(
                    xi2.reshape(1, -1)
                )  # The prediction probability (confidence) list of model 3

                # Find the highest prediction probability among all classes for each ML model
                y_pred_p1 = np.max(p1)
                y_pred_p2 = np.max(p2)
                y_pred_p3 = np.max(p3)

                if (
                    y_pred1 == y_pred2 == y_pred3
                ):  # If the predicted classes of all the three models are the same
                    y_pred = (
                        y_pred1  # Use this predicted class as the final predicted class
                    )

                elif (
                    y_pred1 != y_pred2 != y_pred3
                ):  # If the predicted classes of all the three models are different
                    # For each prediction model, check if the predicted class’s original ML model is the same as its leader model
                    if (
                        model[y_pred1] == m1
                    ):  # If they are the same and the leading model is model 1 (LightGBM)
                        l.append(m1)
                        pred_l.append(y_pred1)  # Save the predicted class
                        pro_l.append(y_pred_p1)  # Save the confidence

                    if (
                        model[y_pred2] == m2
                    ):  # If they are the same and the leading model is model 2 (XGBoost)
                        l.append(m2)
                        pred_l.append(y_pred2)
                        pro_l.append(y_pred_p2)

                    if (
                        model[y_pred3] == m3
                    ):  # If they are the same and the leading model is model 3 (CatBoost)
                        l.append(m3)
                        pred_l.append(y_pred3)
                        pro_l.append(y_pred_p3)

                    if len(l) == 0:  # Avoid empty probability list
                        pro_l = [y_pred_p1, y_pred_p2, y_pred_p3]

                    elif (
                        len(l) == 1
                    ):  # If only one pair of the original model and the leader model for each predicted class is the same
                        y_pred = pred_l[
                            0
                        ]  # Use the predicted class of the leader model as the final prediction class

                    else:  # If no pair or multiple pairs of the original prediction model and the leader model for each predicted class are the same
                        max_p = max(pro_l)  # Find the highest confidence

                        # Use the predicted class with the highest confidence as the final prediction class
                        if max_p == y_pred_p1:
                            y_pred = y_pred1
                        elif max_p == y_pred_p2:
                            y_pred = y_pred2
                        else:
                            y_pred = y_pred3

                else:  # If two predicted classes are the same and the other one is different
                    n = mode(
                        [y_pred1, y_pred2, y_pred3]
                    )  # Find the predicted class with the majority vote
                    y_pred = model[n].predict(
                        xi2.reshape(1, -1)
                    )  # Use the predicted class of the leader model as the final prediction class
                    y_pred = int(y_pred[0])

                yt.append(yi)
                yp.append(y_pred)  # Save the predicted classes for all tested samples
            return yt, yp

        # Implementing LCCDE
        yt, yp = do_LCCDE(X_test, y_test, m1=lg, m2=xg, m3=cb)

        # The performance of the proposed lCCDE model
        print("Accuracy of LCCDE: " + str(accuracy_score(yt, yp)))
        print("Precision of LCCDE: " + str(precision_score(yt, yp, average="weighted")))
        print("Recall of LCCDE: " + str(recall_score(yt, yp, average="weighted")))
        print("Average F1 of LCCDE: " + str(f1_score(yt, yp, average="weighted")))
        print(
            "F1 of LCCDE for each type of attack: "
            + str(f1_score(yt, yp, average=None))
        )

        # Comparison: The F1-scores for each base model
        print("F1 of LightGBM for each type of attack: " + str(lg_f1))
        print("F1 of XGBoost for each type of attack: " + str(xg_f1))
        print("F1 of CatBoost for each type of attack: " + str(cb_f1))

        ######################### END OF CODE FROM LCCDE_IDS_GlobeCom22.ipynb

        accuracy = accuracy_score(yt, yp)
        precision = precision_score(yt, yp, average="weighted")
        recall = recall_score(yt, yp, average="weighted")
        avg_f1 = f1_score(yt, yp, average="weighted")
        f1 = f1_score(yt, yp, average=None)

        end_time = dt.datetime.now()
        # total_duration = time.time() - start_time
        results = {
            "model": self.name,
            "parameters": self.parameters,
            "start_time": str(start_time),
            "end_time": str(end_time),
            # "total_duration": total_duration,
            "results": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "avg_f1": avg_f1,
                "categorical_f1": f1.tolist(),
            },
        }

        return results, None


# TODO
class MTH(Model):
    name = "MTH"
    parameters = {}


# TODO
class TreeBased(Model):
    name = "TreeBased"
    parameters = {}


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
