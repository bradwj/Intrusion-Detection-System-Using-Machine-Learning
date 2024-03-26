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


logger = logging.getLogger("server.engine")


class Model:
    def __init__(self):
        if not hasattr(self, "name"):
            raise NotImplementedError("Model must have a 'name' attribute")
        if not hasattr(self, "parameters"):
            self.parameters = []
        self.parameter_values = {param.name: param.value for param in self.parameters}

    @classmethod
    def get_models(self):
        return [model.to_dict() for model in self.__subclasses__()]

    @classmethod
    def get_parameter_metadata(self):
        return [param.get_metadata() for param in self.parameters]

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
                if not isinstance(value, param_obj.dtype):
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
                param.value = input_parameters[param.name]


class Parameter:
    # TODO: add valid range for values
    def __init__(
        self,
        name: str,
        dtype: type,
        default=None,
        range: tuple = None,
        value=None,
    ):
        self.name = name
        self.dtype = dtype
        self.default = default
        self.range = range
        self.value = value if value is not None else default

    def get_metadata(self):
        return {
            "name": self.name,
            "dtype": str(self.dtype).split("'")[1],
            "default": self.default,
            "range": self.range,
        }

    def to_dict(self):
        return self.get_metadata() | {"value": self.value}


class LCCDE(Model):
    name = "LCCDE"
    parameters = [
        Parameter("num_iterations", int, default=100, range=(0, None)),
        Parameter("learning_rate", float, default=0.1, range=(0.0, None)),
        Parameter("num_leaves", int, default=31, range=(1, 131072)),
        Parameter("max_depth", int, default=-1, range=(0, None)),
        Parameter("min_data_in_leaf", int, default=20, range=(0, None)),
    ]

    # parameter_values is dict with {param.name: param_value} for each param in parameters
    def run(self):
        logger.info(
            f"Running {self.name} model with parameters: {self.parameter_values}"
        )
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

        lg = lgb.LGBMClassifier()
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

        xg = xgb.XGBClassifier()

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

        cb = cbt.CatBoostClassifier(verbose=0, boosting_type="Plain")
        # cb = cbt.CatBoostClassifier()

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
        total_duration = end_time - start_time
        results = {
            "model": self.name,
            "parameters": [
                {"name": k, "value": v} for k, v in self.parameter_values.items()
            ],
            "start_time": str(start_time),
            "end_time": str(end_time),
            "total_duration": total_duration,
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
