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
            "description": "Number of boosting iterations.",
        },
        "learning_rate": {
            "dtype": "float",
            "default": 0.1,
            "range": "(0,inf)",
            "description": "Shrinkage rate.",
        },
        "num_leaves": {
            "dtype": "int",
            "default": 31,
            "range": "(1,131072]",
            "description": "Max number of leaves in one tree.",
        },
        "max_depth": {
            "dtype": "int",
            "default": -1,
            "description": "Limit the max depth for tree model. This is used to deal with over-fitting when data is small. Tree still grows leaf-wise",
        },
        "min_data_in_leaf": {
            "dtype": "int",
            "default": 20,
            "range": "[0,inf)",
            "description": "Minimal number of data in one leaf. Can be used to deal with over-fitting.",
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
        "learning_rate": {
            "dtype": "float",
            "default": 0.3,
            "range": "[0,1]",
            "description": "Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.",
        },
        "gamma": {
            "dtype": "float",
            "default": 0,
            "range": "[0,inf)",
            "description": "Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.",
        },
        "max_depth": {
            "dtype": "int",
            "default": 6,
            "range": "[0,inf)",
            "description": "Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. Exact tree method requires non-zero value.",
        },
        "min_child_weight": {
            "dtype": "int",
            "default": 1,
            "range": "[0,inf)",
            "description": "Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.",
        },
        "max_delta_step": {
            "dtype": "int",
            "default": 0,
            "range": "[0,inf)",
            "description": "Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.",
        },
        "subsample": {
            "dtype": "float",
            "default": 1,
            "range": "(0,1]",
            "description": "Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.",
        },
        "lambda": {
            "dtype": "float",
            "default": 1,
            "range": "[0,inf)",
            "description": "L2 regularization term on weights. Increasing this value will make model more conservative.",
        },
        "alpha": {
            "dtype": "float",
            "default": 0,
            "range": "[0,inf)",
            "description": "L1 regularization term on weights. Increasing this value will make model more conservative.",
        },
        "tree_method": {
            "dtype": "str",
            "default": "auto",
            "choices": ["auto", "exact", "approx", "hist"],
            "description": "The tree construction algorithm used in XGBoost.",
        },
        "max_leaves": {
            "dtype": "int",
            "default": 0,
            "range": "[0,inf)",
            "description": "Maximum number of nodes to be added. Not used by exact tree method.",
        },
    },
    CATBOOST_CLASSIFIER: {
        "iterations": {
            "dtype": "int",
            "default": 1000,
            "range": "[0,inf)",
            "description": "The maximum number of trees that can be built when solving machine learning problems. When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter.",
        },
        "learning_rate": {
            "dtype": "float",
            "default": 0.03,
            "range": "(0,inf)",
            "description": "The learning rate. Used for reducing the gradient step.",
        },
        "sampling_frequency": {
            "dtype": "str",
            "default": "PerTree",
            "choices": ["PerTree", "PerTreeLevel", "PerTreeLevelAndFeature"],
            "description": "Frequency to sample weights and objects when building trees.",
        },
        "grow_policy": {
            "dtype": "str",
            "default": "SymmetricTree",
            "choices": ["SymmetricTree", "Depthwise", "Lossguide"],
            "description": "Tree growth policy. SymmetricTree -- tree is built level by level until the specified depth is reached. On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric. Depthwise — A tree is built level by level until the specified depth is reached. On each iteration, all non-terminal leaves from the last tree level are split. Each leaf is split by condition with the best loss improvement. Lossguide — A tree is built leaf by leaf until the specified maximum number of leaves is reached. On each iteration, non-terminal leaf with the best loss improvement is split.",
        },
        "min_data_in_leaf": {
            "dtype": "int",
            "default": 1,
            "range": "[0,inf)",
            "description": "The minimum number of training samples in a leaf. CatBoost does not search for new splits in leaves with samples count less than the specified value. Can be used only with the Lossguide and Depthwise growing policies.",
        },
        "max_leaves": {
            "dtype": "int",
            "default": 31,
            "range": "[0,inf)",
            "description": "The maximum number of leafs in the resulting tree. Can be used only with the Lossguide growing policy.",
        },
        "boosting_type": {
            "dtype": "str",
            "default": "Plain",
            "choices": ["Plain", "Ordered"],
            "description": "Boosting scheme. Ordered — Usually provides better quality on small datasets, but it may be slower than the Plain scheme. Plain — The classic gradient boosting scheme.",
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
        total_duration = (end_time - start_time).total_seconds()
        results = {
            "model": self.name,
            "parameters": self.parameters,
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
    parameters = {}


# TODO
class TreeBased(Model):
    name = "TreeBased"
    parameters = {}
