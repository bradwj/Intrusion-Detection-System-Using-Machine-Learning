import time
import logging
from enum import Enum

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
MINIBATCH_KMEANS = "mini_batch_kmeans"
DECISIONTREE_CLASSIFIER = "decision_tree_classifier"
RANDOMFOREST_CLASSIFIER = "random_forest_classifier"
EXTRATREES_CLASSIFIER = "extra_trees_classifier"

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
        "n_estimators": {
            "dtype": "int",
            "default": 100,
            "range": "[0,inf)",
            "description": "Determines the number of boosting rounds or trees to build.",
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
    MINIBATCH_KMEANS: {
        "n_clusters": {
            "dtype": "int",
            "default": 8,
            "range": "[0,inf)",
            "description": "The number of clusters to form as well as the number of centroids to generate.",
        },
        "random_state": {
            "dtype": "int",
            "default": 0,
            "range": "[0,inf)",
            "description": "Determines random number generation for centroid initialization.",
        },
        "max_iter": {
            "dtype": "int",
            "default": 100,
            "range": "[0,inf)",
            "description": "Maximum number of iterations over the complete dataset before stopping independently of any early stopping criterion heuristics.",
        },
        "batch_size": {
            "dtype": "int",
            "default": 1024,
            "range": "[0,inf)",
            "description": "Size of the mini batches. For faster computations, you can set the batch_size greater than 256 * number of cores to enable parallelism on all cores.",
        },
    },
    DECISIONTREE_CLASSIFIER: {
        "criterion": {
            "dtype": "str",
            "default": "gini",
            "choices": ["gini", "entropy", "log_loss"],
            "description": "The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'log_loss' and 'entropy' both for the Shannon information gain, see Mathematical formulation.",
        },
        "splitter": {
            "dtype": "str",
            "default": "best",
            "choices": ["best", "random"],
            "description": "The strategy used to choose the split at each node. Supported strategies are 'best' to choose the best split and 'random' to choose the best random split.",
        },
        "max_depth": {
            "dtype": "int",
            "default": "None",
            "range": "[0,inf)",
            "description": "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
        },
        "min_samples_split": {
            "dtype": "int",
            "default": "2",
            "range": "[2,inf)",
            "description": "The minimum number of samples required to split an internal node:",
        },
        "min_samples_leaf": {
            "dtype": "int",
            "default": "1",
            "range": "[1,inf)",
            "description": "The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        },
        "min_weight_fraction_leaf": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0.0,inf)",
            "description": "The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.",
        },
        "random_state": {
            "dtype": "int",
            "default": "1",
            "range": "[1,inf)",
            "description": "Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to 'best'.",
        },
        "max_leaf_nodes": {
            "dtype": "int",
            "default": "None",
            "range": "[0,inf)",
            "description": "Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
        },
        "min_impurity_decrease": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0.0,inf)",
            "description": "Description from the HTML.",
        },
        "ccp_alpha": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0.0,inf)",
            "description": "Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.",
        },
    },
    RANDOMFOREST_CLASSIFIER: {
        "n_estimators": {
            "dtype": "int",
            "default": "100",
            "range": "[10,inf)",
            "description": "The number of trees in the forest.",
        },
        "criterion": {
            "dtype": "str",
            "default": "gini",
            "choices": ["gini", "entropy", "log_loss"],
            "description": "The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'log_loss' and 'entropy' both for the Shannon information gain, see Mathematical formulation.",
        },
        "max_depth": {
            "dtype": "int",
            "default": "None",
            "range": "[0,inf)",
            "description": "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
        },
        "min_samples_split": {
            "dtype": "int",
            "default": "2",
            "range": "[2,inf)",
            "description": "The minimum number of samples required to split an internal node:",
        },
        "min_samples_leaf": {
            "dtype": "int",
            "default": "1",
            "range": "[1,inf)",
            "description": "The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        },
        "min_weight_fraction_leaf": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0,inf)",
            "description": "he minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.",
        },
        "max_leaf_nodes": {
            "dtype": "int",
            "default": "None",
            "range": "[0,inf)",
            "description": "Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
        },
        "min_impurity_decrease": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0.0,inf)",
            "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
        },
        "random_state": {
            "dtype": "int",
            "default": "1",
            "range": "[1,inf)",
            "description": "Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to 'best'.",
        },
        "ccp_alpha": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0.0,inf)",
            "description": "Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.",
        },
    },
    EXTRATREES_CLASSIFIER: {
        "n_estimators": {
            "dtype": "int",
            "default": "100",
            "range": "[10,inf)",
            "description": "The number of trees in the forest.",
        },
        "criterion": {
            "dtype": "str",
            "default": "gini",
            "choices": ["gini", "entropy", "log_loss"],
            "description": "The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'log_loss' and 'entropy' both for the Shannon information gain, see Mathematical formulation.",
        },
        "max_depth": {
            "dtype": "int",
            "default": "None",
            "range": "[0,inf)",
            "description": "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.",
        },
        "min_samples_split": {
            "dtype": "int",
            "default": "2",
            "range": "[2,inf)",
            "description": "The minimum number of samples required to split an internal node:",
        },
        "min_samples_leaf": {
            "dtype": "int",
            "default": "1",
            "range": "[1,inf)",
            "description": "The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.",
        },
        "min_weight_fraction_leaf": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0,inf)",
            "description": "he minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.",
        },
        "max_leaf_nodes": {
            "dtype": "int",
            "default": "None",
            "range": "[0,inf)",
            "description": "Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.",
        },
        "min_impurity_decrease": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0.0,inf)",
            "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
        },
        "random_state": {
            "dtype": "int",
            "default": "1",
            "range": "[1,inf)",
            "description": "Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to 'best'.",
        },
        "ccp_alpha": {
            "dtype": "float",
            "default": "0.0",
            "range": "[0.0,inf)",
            "description": "Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.",
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

    def set_dataset(self, dataset):
        self.dataset = dataset

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
        import datetime as datetime

        start_time = datetime.datetime.now()

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

        end_time = datetime.datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        results = {
            "model": self.name,
            "dataset": self.dataset,
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


class MTH(Model):
    name = "MTH"
    parameters = {
        MINIBATCH_KMEANS: {"n_clusters": 1000, "random_state": 0},
        DECISIONTREE_CLASSIFIER: {"random_state": 0},
        RANDOMFOREST_CLASSIFIER: {"random_state": 0},
        EXTRATREES_CLASSIFIER: {"random_state": 0},
        XGBOOST_CLASSIFIER: {"n_estimators": 10},
    }

    def run(self):
        logger.info(f"Running {self.name} model with parameters: {self.parameters}")
        import datetime as datetime

        start_time = datetime.datatetimetime.now()

        ###### Code from MTH_IDS_IoTJ.ipynb
        import warnings

        warnings.filterwarnings("ignore")

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            accuracy_score,
            precision_recall_fscore_support,
        )
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.tree import DecisionTreeClassifier
        import xgboost as xgb
        from xgboost import plot_importance

        # Read dataset
        # df = pd.read_csv("./data/CICIDS2017.csv")
        df = pd.read_csv("./data/CICIDS2017_sample.csv")

        # Z-score normalization
        features = df.dtypes[df.dtypes != "object"].index
        df[features] = df[features].apply(lambda x: (x - x.mean()) / (x.std()))
        # Fill empty values by 0
        df = df.fillna(0)

        labelencoder = LabelEncoder()
        df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
        # retain the minority class instances and sample the majority class instances
        df_minor = df[(df["Label"] == 6) | (df["Label"] == 1) | (df["Label"] == 4)]
        df_major = df.drop(df_minor.index)

        X = df_major.drop(["Label"], axis=1)
        y = df_major.iloc[:, -1].values.reshape(-1, 1)
        y = np.ravel(y)

        logger.info("Performing k-means clustering")
        # use k-means to cluster the data samples and select a proportion of data from each cluster
        from sklearn.cluster import MiniBatchKMeans

        # kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)
        kmeans = MiniBatchKMeans(**self.parameters[MINIBATCH_KMEANS]).fit(X)

        klabel = kmeans.labels_
        df_major["klabel"] = klabel

        cols = list(df_major)
        cols.insert(78, cols.pop(cols.index("Label")))
        df_major = df_major.loc[:, cols]

        def typicalSampling(group):
            name = group.name
            frac = 0.008
            return group.sample(frac=frac)

        result = df_major.groupby("klabel", group_keys=False).apply(typicalSampling)

        result = result.drop(["klabel"], axis=1)
        # result = result.append(df_minor)
        pd.concat([result, df_minor], axis=0)

        # result.to_csv("./data/CICIDS2017_sample_km.csv", index=0)

        # Read the sampled dataset
        df = pd.read_csv("./data/CICIDS2017_sample_km.csv")
        X = df.drop(["Label"], axis=1).values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        y = np.ravel(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y
        )

        logger.info("Performing feature selection")
        from sklearn.feature_selection import mutual_info_classif

        importances = mutual_info_classif(X_train, y_train)

        # calculate the sum of importance scores
        f_list = sorted(
            zip(map(lambda x: round(x, 4), importances), features), reverse=True
        )
        Sum = 0
        fs = []
        for i in range(0, len(f_list)):
            Sum = Sum + f_list[i][0]
            fs.append(f_list[i][1])

        # select the important features from top to bottom until the accumulated importance reaches 90%
        f_list2 = sorted(
            zip(map(lambda x: round(x, 4), importances / Sum), features), reverse=True
        )
        Sum2 = 0
        fs = []
        for i in range(0, len(f_list2)):
            Sum2 = Sum2 + f_list2[i][0]
            fs.append(f_list2[i][1])
            if Sum2 >= 0.9:
                break

        X_fs = df[fs].values

        logger.info(
            "Performing feature selection by Fast Correlation Based Filter (FCBF)"
        )
        # feature selection by Fast Correlation Based Filter (FCBF)
        from FCBF_module.FCBF_module import FCBF, FCBFK, FCBFiP, get_i

        fcbf = FCBFK(k=20)
        # fcbf.fit(X_fs, y)

        X_fss = fcbf.fit_transform(X_fs, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_fss, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y
        )

        from imblearn.over_sampling import SMOTE

        smote = SMOTE(n_jobs=-1, sampling_strategy={2: 1000, 4: 1000})

        X_train, y_train = smote.fit_resample(X_train, y_train)

        # train four base learners: decision tree. random forest, extra trees, XGBoost

        logger.info("Training base XGB classifier")
        # xg = xgb.XGBClassifier(n_estimators=10)
        xg = xgb.XGBClassifier(**self.parameters[XGBOOST_CLASSIFIER])
        xg.fit(X_train, y_train)
        xg_score = xg.score(X_test, y_test)
        y_predict = xg.predict(X_test)
        y_true = y_test
        print("Accuracy of XGBoost: " + str(xg_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of XGBoost: " + (str(precision)))
        print("Recall of XGBoost: " + (str(recall)))
        print("F1-score of XGBoost: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        def objective(params):
            params = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "learning_rate": abs(float(params["learning_rate"])),
            }
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            return {"loss": -score, "status": STATUS_OK}

        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
            "max_depth": hp.quniform("max_depth", 4, 100, 1),
            "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
        print("XGBoost: Hyperopt estimated optimum {}".format(best))

        # best={'learning_rate': 0.7340229699980686, 'n_estimators': 70.0, 'max_depth': 14.0}
        # uses learning rate, n_estimators, and max_depth from hyperopt
        xg = xgb.XGBClassifier(
            learning_rate=0.7340229699980686, n_estimators=70, max_depth=14
        )
        xg.fit(X_train, y_train)
        xg_score = xg.score(X_test, y_test)
        y_predict = xg.predict(X_test)
        y_true = y_test
        print("Accuracy of XGBoost: " + str(xg_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of XGBoost: " + (str(precision)))
        print("Recall of XGBoost: " + (str(recall)))
        print("F1-score of XGBoost: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        xg_train = xg.predict(X_train)
        xg_test = xg.predict(X_test)

        logger.info("RFC")
        # rf = RandomForestClassifier(random_state=0)
        rf = RandomForestClassifier(**self.parameters[RANDOMFOREST_CLASSIFIER])
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        y_predict = rf.predict(X_test)
        y_true = y_test
        print("Accuracy of RF: " + str(rf_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of RF: " + (str(precision)))
        print("Recall of RF: " + (str(recall)))
        print("F1-score of RF: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        # Hyperparameter optimization of random forest
        from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        # Define the objective function
        def objective(params):
            params = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "max_features": int(params["max_features"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "criterion": str(params["criterion"]),
            }
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            return {"loss": -score, "status": STATUS_OK}

        # Define the hyperparameter configuration space
        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, 20, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
        print("Random Forest: Hyperopt estimated optimum {}".format(best))

        rf_hpo = RandomForestClassifier(
            n_estimators=71,
            min_samples_leaf=1,
            max_depth=46,
            min_samples_split=9,
            max_features=20,
            criterion="entropy",
        )
        rf_hpo.fit(X_train, y_train)
        rf_score = rf_hpo.score(X_test, y_test)
        y_predict = rf_hpo.predict(X_test)
        y_true = y_test
        print("Accuracy of RF: " + str(rf_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of RF: " + (str(precision)))
        print("Recall of RF: " + (str(recall)))
        print("F1-score of RF: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        rf_train = rf_hpo.predict(X_train)
        rf_test = rf_hpo.predict(X_test)

        logger.info("DTC")
        # dt = DecisionTreeClassifier(random_state=0)
        dt = DecisionTreeClassifier(**self.parameters[DECISIONTREE_CLASSIFIER])
        dt.fit(X_train, y_train)
        dt_score = dt.score(X_test, y_test)
        y_predict = dt.predict(X_test)
        y_true = y_test
        print("Accuracy of DT: " + str(dt_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of DT: " + (str(precision)))
        print("Recall of DT: " + (str(recall)))
        print("F1-score of DT: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        # Hyperparameter optimization of decision tree
        from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        # Define the objective function
        def objective(params):
            params = {
                "max_depth": int(params["max_depth"]),
                "max_features": int(params["max_features"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "criterion": str(params["criterion"]),
            }
            clf = DecisionTreeClassifier(**params)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            return {"loss": -score, "status": STATUS_OK}

        # Define the hyperparameter configuration space
        space = {
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, 20, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)
        print("Decision tree: Hyperopt estimated optimum {}".format(best))

        dt_hpo = DecisionTreeClassifier(
            min_samples_leaf=2,
            max_depth=47,
            min_samples_split=3,
            max_features=19,
            criterion="gini",
        )
        dt_hpo.fit(X_train, y_train)
        dt_score = dt_hpo.score(X_test, y_test)
        y_predict = dt_hpo.predict(X_test)
        y_true = y_test
        print("Accuracy of DT: " + str(dt_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of DT: " + (str(precision)))
        print("Recall of DT: " + (str(recall)))
        print("F1-score of DT: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        dt_train = dt_hpo.predict(X_train)
        dt_test = dt_hpo.predict(X_test)

        logger.info("ETC")
        # et = ExtraTreesClassifier(random_state=0)
        et = ExtraTreesClassifier(**self.parameters[EXTRATREES_CLASSIFIER])
        et.fit(X_train, y_train)
        et_score = et.score(X_test, y_test)
        y_predict = et.predict(X_test)
        y_true = y_test
        print("Accuracy of ET: " + str(et_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of ET: " + (str(precision)))
        print("Recall of ET: " + (str(recall)))
        print("F1-score of ET: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        # Hyperparameter optimization of extra trees
        from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        # Define the objective function
        def objective(params):
            params = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "max_features": int(params["max_features"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "criterion": str(params["criterion"]),
            }
            clf = ExtraTreesClassifier(**params)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            return {"loss": -score, "status": STATUS_OK}

        # Define the hyperparameter configuration space
        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, 20, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
        print("Extra Trees: Hyperopt estimated optimum {}".format(best))

        et_hpo = ExtraTreesClassifier(
            n_estimators=53,
            min_samples_leaf=1,
            max_depth=31,
            min_samples_split=5,
            max_features=20,
            criterion="entropy",
        )
        et_hpo.fit(X_train, y_train)
        et_score = et_hpo.score(X_test, y_test)
        y_predict = et_hpo.predict(X_test)
        y_true = y_test
        print("Accuracy of ET: " + str(et_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of ET: " + (str(precision)))
        print("Recall of ET: " + (str(recall)))
        print("F1-score of ET: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        et_train = et_hpo.predict(X_train)
        et_test = et_hpo.predict(X_test)

        # apply stacking
        base_predictions_train = pd.DataFrame(
            {
                "DecisionTree": dt_train.ravel(),
                "RandomForest": rf_train.ravel(),
                "ExtraTrees": et_train.ravel(),
                "XgBoost": xg_train.ravel(),
            }
        )
        base_predictions_train.head(5)

        dt_train = dt_train.reshape(-1, 1)
        et_train = et_train.reshape(-1, 1)
        rf_train = rf_train.reshape(-1, 1)
        xg_train = xg_train.reshape(-1, 1)
        dt_test = dt_test.reshape(-1, 1)
        et_test = et_test.reshape(-1, 1)
        rf_test = rf_test.reshape(-1, 1)
        xg_test = xg_test.reshape(-1, 1)

        x_train = np.concatenate((dt_train, et_train, rf_train, xg_train), axis=1)
        x_test = np.concatenate((dt_test, et_test, rf_test, xg_test), axis=1)

        stk = xgb.XGBClassifier().fit(x_train, y_train)
        y_predict = stk.predict(x_test)
        y_true = y_test
        stk_score = accuracy_score(y_true, y_predict)
        print("Accuracy of Stacking: " + str(stk_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of Stacking: " + (str(precision)))
        print("Recall of Stacking: " + (str(recall)))
        print("F1-score of Stacking: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        def objective(params):
            params = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "learning_rate": abs(float(params["learning_rate"])),
            }
            clf = xgb.XGBClassifier(**params)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            score = accuracy_score(y_test, y_pred)

            return {"loss": -score, "status": STATUS_OK}

        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
            "max_depth": hp.quniform("max_depth", 4, 100, 1),
            "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
        print("XGBoost: Hyperopt estimated optimum {}".format(best))

        xg = xgb.XGBClassifier(
            learning_rate=0.19229249758051492, n_estimators=30, max_depth=36
        )
        xg.fit(x_train, y_train)
        xg_score = xg.score(x_test, y_test)
        y_predict = xg.predict(x_test)
        y_true = y_test
        print("Accuracy of XGBoost: " + str(xg_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of XGBoost: " + (str(precision)))
        print("Recall of XGBoost: " + (str(recall)))
        print("F1-score of XGBoost: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        logger.info("Anomaly-based IDS")
        # anomaly-based IDS
        df = pd.read_csv("./data/CICIDS2017_sample_km.csv")
        df1 = df[df["Label"] != 5]
        df1["Label"][df1["Label"] > 0] = 1
        df1.to_csv("./data/CICIDS2017_sample_km_without_portscan.csv", index=0)

        df2 = df[df["Label"] == 5]
        df2["Label"][df2["Label"] == 5] = 1
        df2.to_csv("./data/CICIDS2017_sample_km_portscan.csv", index=0)

        df1 = pd.read_csv("./data/CICIDS2017_sample_km_without_portscan.csv")
        df2 = pd.read_csv("./data/CICIDS2017_sample_km_portscan.csv")

        features = df1.drop(["Label"], axis=1).dtypes[df1.dtypes != "object"].index
        df1[features] = df1[features].apply(lambda x: (x - x.mean()) / (x.std()))
        df2[features] = df2[features].apply(lambda x: (x - x.mean()) / (x.std()))
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)

        df2p = df1[df1["Label"] == 0]
        df2pp = df2p.sample(
            n=None,
            frac=1255 / 18225,
            replace=False,
            weights=None,
            random_state=None,
            axis=0,
        )
        df2 = pd.concat([df2, df2pp])

        df = pd.concat([df1, df2])

        X = df.drop(["Label"], axis=1).values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        y = np.ravel(y)
        # pd.Series(y).value_counts()

        # feature engineering (IG, FCBF, and KPCA)

        # feature selection by information gain (IG)
        from sklearn.feature_selection import mutual_info_classif

        importances = mutual_info_classif(X, y)

        # calculate the sum of importance scores
        f_list = sorted(
            zip(map(lambda x: round(x, 4), importances), features), reverse=True
        )
        Sum = 0
        fs = []
        for i in range(0, len(f_list)):
            Sum = Sum + f_list[i][0]
            fs.append(f_list[i][1])

        # select the important features from top to bottom until the accumulated importance reaches 90%
        f_list2 = sorted(
            zip(map(lambda x: round(x, 4), importances / Sum), features), reverse=True
        )
        Sum2 = 0
        fs = []
        for i in range(0, len(f_list2)):
            Sum2 = Sum2 + f_list2[i][0]
            fs.append(f_list2[i][1])
            if Sum2 >= 0.9:
                break

        X_fs = df[fs].values

        from FCBF_module.FCBF_module import FCBF, FCBFK, FCBFiP, get_i

        fcbf = FCBFK(k=20)
        # fcbf.fit(X_fs, y)

        X_fss = fcbf.fit_transform(X_fs, y)

        from sklearn.decomposition import KernelPCA

        kpca = KernelPCA(n_components=10, kernel="rbf")
        kpca.fit(X_fss, y)
        X_kpca = kpca.transform(X_fss)

        # from sklearn.decomposition import PCA
        # kpca = PCA(n_components = 10)
        # kpca.fit(X_fss, y)
        # X_kpca = kpca.transform(X_fss)

        X_train = X_kpca[: len(df1)]
        y_train = y[: len(df1)]
        X_test = X_kpca[len(df1) :]
        y_test = y[len(df1) :]

        from imblearn.over_sampling import SMOTE

        smote = SMOTE(n_jobs=-1, sampling_strategy={1: 18225})
        X_train, y_train = smote.fit_resample(X_train, y_train)

        from sklearn.cluster import KMeans
        from sklearn.cluster import DBSCAN, MeanShift
        from sklearn.cluster import (
            SpectralClustering,
            AgglomerativeClustering,
            AffinityPropagation,
            Birch,
            MiniBatchKMeans,
            MeanShift,
        )
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
        from sklearn.metrics import classification_report
        from sklearn import metrics

        def CL_kmeans(X_train, X_test, y_train, y_test, n, b=100):
            km_cluster = MiniBatchKMeans(n_clusters=n, batch_size=b)
            result = km_cluster.fit_predict(X_train)
            result2 = km_cluster.predict(X_test)

            count = 0
            a = np.zeros(n)
            b = np.zeros(n)
            for v in range(0, n):
                for i in range(0, len(y_train)):
                    if result[i] == v:
                        if y_train[i] == 1:
                            a[v] = a[v] + 1
                        else:
                            b[v] = b[v] + 1
            list1 = []
            list2 = []
            for v in range(0, n):
                if a[v] <= b[v]:
                    list1.append(v)
                else:
                    list2.append(v)
            for v in range(0, len(y_test)):
                if result2[v] in list1:
                    result2[v] = 0
                elif result2[v] in list2:
                    result2[v] = 1
                else:
                    print("-1")
            print(classification_report(y_test, result2))
            cm = confusion_matrix(y_test, result2)
            accuracy = metrics.accuracy_score(y_test, result2)
            precision = metrics.precision_score(y_test, result2, average="weighted")
            recall = metrics.recall_score(y_test, result2, average="weighted")
            avg_f1 = metrics.f1_score(y_test, result2, average="weighted")
            f1 = metrics.f1_score(y_test, result2, average=None)
            print(str(accuracy))
            print(cm)
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "avg_f1": avg_f1,
                "categorical_f1": f1.tolist(),
            }

        CL_kmeans(X_train, X_test, y_train, y_test, 8)

        # Hyperparameter optimization by BO-GP
        logger.info("Hyperparameter optimization of CL-k-means by BO-GP")
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
        from sklearn import metrics

        space = [Integer(2, 50, name="n_clusters")]

        @use_named_args(space)
        def objective(**params):
            km_cluster = MiniBatchKMeans(batch_size=100, **params)
            n = params["n_clusters"]

            result = km_cluster.fit_predict(X_train)
            result2 = km_cluster.predict(X_test)

            count = 0
            a = np.zeros(n)
            b = np.zeros(n)
            for v in range(0, n):
                for i in range(0, len(y_train)):
                    if result[i] == v:
                        if y_train[i] == 1:
                            a[v] = a[v] + 1
                        else:
                            b[v] = b[v] + 1
            list1 = []
            list2 = []
            for v in range(0, n):
                if a[v] <= b[v]:
                    list1.append(v)
                else:
                    list2.append(v)
            for v in range(0, len(y_test)):
                if result2[v] in list1:
                    result2[v] = 0
                elif result2[v] in list2:
                    result2[v] = 1
                else:
                    print("-1")
            cm = metrics.accuracy_score(y_test, result2)
            print(str(n) + " " + str(cm))
            return 1 - cm

        from skopt import gp_minimize
        import time

        t1 = time.time()
        res_gp = gp_minimize(objective, space, n_calls=20, random_state=0)
        t2 = time.time()
        print(t2 - t1)
        print("Best score=%.4f" % (1 - res_gp.fun))
        print("""Best parameters: n_clusters=%d""" % (res_gp.x[0]))

        # Hyperparameter optimization by BO-TPE
        logger.info("Hyperparameter optimization by BO-TPE")
        from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.cluster import MiniBatchKMeans
        from sklearn import metrics

        def objective(params):
            params = {
                "n_clusters": int(params["n_clusters"]),
            }
            km_cluster = MiniBatchKMeans(batch_size=100, **params)
            n = params["n_clusters"]

            from sklearn.decomposition import PCA

            result = km_cluster.fit_predict(X_train)
            result2 = km_cluster.predict(X_test)

            count = 0
            a = np.zeros(n)
            b = np.zeros(n)
            for v in range(0, n):
                for i in range(0, len(y_train)):
                    if result[i] == v:
                        if y_train[i] == 1:
                            a[v] = a[v] + 1
                        else:
                            b[v] = b[v] + 1
            list1 = []
            list2 = []
            for v in range(0, n):
                if a[v] <= b[v]:
                    list1.append(v)
                else:
                    list2.append(v)
            for v in range(0, len(y_test)):
                if result2[v] in list1:
                    result2[v] = 0
                elif result2[v] in list2:
                    result2[v] = 1
                else:
                    print("-1")
            score = metrics.accuracy_score(y_test, result2)
            print(str(params["n_clusters"]) + " " + str(score))
            return {"loss": 1 - score, "status": STATUS_OK}

        space = {
            "n_clusters": hp.quniform("n_clusters", 2, 50, 1),
        }

        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
        print("Random Forest: Hyperopt estimated optimum {}".format(best))

        res = CL_kmeans(X_train, X_test, y_train, y_test, 16)

        end_time = datetime.datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        results = {
            "model": self.name,
            "dataset": self.dataset,
            "parameters": self.parameters,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "total_duration": total_duration,
            "results": res,
        }

        return results, None


class TreeBased(Model):
    name = "TreeBased"
    parameters = {
        DECISIONTREE_CLASSIFIER: {"random_state": 0},
        RANDOMFOREST_CLASSIFIER: {"random_state": 0},
        EXTRATREES_CLASSIFIER: {"random_state": 0},
        XGBOOST_CLASSIFIER: {"n_estimators": 10},
    }

    def run(self):
        logger.info(f"Running {self.name} model with parameters: {self.parameters}")
        import datetime as datetime

        start_time = datetime.datetime.now()

        ###### Code from Tree-based_IDS_GlobeCom19.ipynb
        import warnings

        warnings.filterwarnings("ignore")

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            accuracy_score,
            precision_recall_fscore_support,
            f1_score,
        )
        from sklearn.metrics import f1_score
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.tree import DecisionTreeClassifier
        import xgboost as xgb
        from xgboost import plot_importance

        df = pd.read_csv("./data/CICIDS2017_sample.csv")

        # Min-max normalization
        numeric_features = df.dtypes[df.dtypes != "object"].index
        df[numeric_features] = df[numeric_features].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        # Fill empty values by 0
        df = df.fillna(0)

        labelencoder = LabelEncoder()
        df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
        X = df.drop(["Label"], axis=1).values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        y = np.ravel(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y
        )

        from imblearn.over_sampling import SMOTE

        smote = SMOTE(
            n_jobs=-1, sampling_strategy={4: 1500}
        )  # Create 1500 samples for the minority class "4"

        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Decision tree training and prediction
        # dt = DecisionTreeClassifier(random_state=0)
        dt = DecisionTreeClassifier(**self.parameters[DECISIONTREE_CLASSIFIER])
        dt.fit(X_train, y_train)
        dt_score = dt.score(X_test, y_test)
        y_predict = dt.predict(X_test)
        y_true = y_test
        print("Accuracy of DT: " + str(dt_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of DT: " + (str(precision)))
        print("Recall of DT: " + (str(recall)))
        print("F1-score of DT: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        dt_train = dt.predict(X_train)
        dt_test = dt.predict(X_test)

        # Random Forest training and prediction
        # rf = RandomForestClassifier(random_state=0)
        rf = RandomForestClassifier(**self.parameters[RANDOMFOREST_CLASSIFIER])
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        y_predict = rf.predict(X_test)
        y_true = y_test
        print("Accuracy of RF: " + str(rf_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of RF: " + (str(precision)))
        print("Recall of RF: " + (str(recall)))
        print("F1-score of RF: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        rf_train = rf.predict(X_train)
        rf_test = rf.predict(X_test)

        # Extra trees training and prediction
        # et = ExtraTreesClassifier(random_state=0)
        et = ExtraTreesClassifier(**self.parameters[EXTRATREES_CLASSIFIER])
        et.fit(X_train, y_train)
        et_score = et.score(X_test, y_test)
        y_predict = et.predict(X_test)
        y_true = y_test
        print("Accuracy of ET: " + str(et_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of ET: " + (str(precision)))
        print("Recall of ET: " + (str(recall)))
        print("F1-score of ET: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        et_train = et.predict(X_train)
        et_test = et.predict(X_test)

        # XGboost training and prediction
        # xg = xgb.XGBClassifier(n_estimators=10)
        xg = xgb.XGBClassifier(**self.parameters[XGBOOST_CLASSIFIER])
        xg.fit(X_train, y_train)
        xg_score = xg.score(X_test, y_test)
        y_predict = xg.predict(X_test)
        y_true = y_test
        print("Accuracy of XGBoost: " + str(xg_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of XGBoost: " + (str(precision)))
        print("Recall of XGBoost: " + (str(recall)))
        print("F1-score of XGBoost: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        xg_train = xg.predict(X_train)
        xg_test = xg.predict(X_test)

        # Stacking model construction (ensemble for 4 base learners)

        # Use the outputs of 4 base models to construct a new ensemble model
        base_predictions_train = pd.DataFrame(
            {
                "DecisionTree": dt_train.ravel(),
                "RandomForest": rf_train.ravel(),
                "ExtraTrees": et_train.ravel(),
                "XgBoost": xg_train.ravel(),
            }
        )
        base_predictions_train.head(5)

        dt_train = dt_train.reshape(-1, 1)
        et_train = et_train.reshape(-1, 1)
        rf_train = rf_train.reshape(-1, 1)
        xg_train = xg_train.reshape(-1, 1)
        dt_test = dt_test.reshape(-1, 1)
        et_test = et_test.reshape(-1, 1)
        rf_test = rf_test.reshape(-1, 1)
        xg_test = xg_test.reshape(-1, 1)

        x_train = np.concatenate((dt_train, et_train, rf_train, xg_train), axis=1)
        x_test = np.concatenate((dt_test, et_test, rf_test, xg_test), axis=1)

        stk = xgb.XGBClassifier().fit(x_train, y_train)

        y_predict = stk.predict(x_test)
        y_true = y_test
        stk_score = accuracy_score(y_true, y_predict)
        print("Accuracy of Stacking: " + str(stk_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of Stacking: " + (str(precision)))
        print("Recall of Stacking: " + (str(recall)))
        print("F1-score of Stacking: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        # Feature selection
        # Save the feature importance lists generated by four tree-based algorithms
        dt_feature = dt.feature_importances_
        rf_feature = rf.feature_importances_
        et_feature = et.feature_importances_
        xgb_feature = xg.feature_importances_

        # calculate the average importance value of each feature
        avg_feature = (dt_feature + rf_feature + et_feature + xgb_feature) / 4

        feature = (df.drop(["Label"], axis=1)).columns.values
        print("Features sorted by their score:")
        print(
            sorted(zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True)
        )

        f_list = sorted(
            zip(map(lambda x: round(x, 4), avg_feature), feature), reverse=True
        )

        # Select the important features from top-importance to bottom-importance until the accumulated importance reaches 0.9 (out of 1)
        Sum = 0
        fs = []
        for i in range(0, len(f_list)):
            Sum = Sum + f_list[i][0]
            fs.append(f_list[i][1])
            if Sum >= 0.9:
                break

        X_fs = df[fs].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_fs, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y
        )

        from imblearn.over_sampling import SMOTE

        smote = SMOTE(n_jobs=-1, sampling_strategy={4: 1500})

        X_train, y_train = smote.fit_resample(X_train, y_train)

        # dt = DecisionTreeClassifier(random_state=0)
        dt = DecisionTreeClassifier(**self.parameters[DECISIONTREE_CLASSIFIER])
        dt.fit(X_train, y_train)
        dt_score = dt.score(X_test, y_test)
        y_predict = dt.predict(X_test)
        y_true = y_test
        print("Accuracy of DT: " + str(dt_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of DT: " + (str(precision)))
        print("Recall of DT: " + (str(recall)))
        print("F1-score of DT: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        dt_train = dt.predict(X_train)
        dt_test = dt.predict(X_test)

        # rf = RandomForestClassifier(random_state=0)
        rf = RandomForestClassifier(**self.parameters[RANDOMFOREST_CLASSIFIER])
        rf.fit(
            X_train, y_train
        )  # modelin veri üzerinde öğrenmesi fit fonksiyonuyla yapılıyor
        rf_score = rf.score(X_test, y_test)
        y_predict = rf.predict(X_test)
        y_true = y_test
        print("Accuracy of RF: " + str(rf_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of RF: " + (str(precision)))
        print("Recall of RF: " + (str(recall)))
        print("F1-score of RF: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        rf_train = rf.predict(X_train)
        rf_test = rf.predict(X_test)

        # et = ExtraTreesClassifier(random_state=0)
        et = ExtraTreesClassifier(**self.parameters[EXTRATREES_CLASSIFIER])
        et.fit(X_train, y_train)
        et_score = et.score(X_test, y_test)
        y_predict = et.predict(X_test)
        y_true = y_test
        print("Accuracy of ET: " + str(et_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of ET: " + (str(precision)))
        print("Recall of ET: " + (str(recall)))
        print("F1-score of ET: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        et_train = et.predict(X_train)
        et_test = et.predict(X_test)

        # xg = xgb.XGBClassifier(n_estimators=10)
        xg = xgb.XGBClassifier(**self.parameters[XGBOOST_CLASSIFIER])
        xg.fit(X_train, y_train)
        xg_score = xg.score(X_test, y_test)
        y_predict = xg.predict(X_test)
        y_true = y_test
        print("Accuracy of XGBoost: " + str(xg_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        print("Precision of XGBoost: " + (str(precision)))
        print("Recall of XGBoost: " + (str(recall)))
        print("F1-score of XGBoost: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        xg_train = xg.predict(X_train)
        xg_test = xg.predict(X_test)

        base_predictions_train = pd.DataFrame(
            {
                "DecisionTree": dt_train.ravel(),
                "RandomForest": rf_train.ravel(),
                "ExtraTrees": et_train.ravel(),
                "XgBoost": xg_train.ravel(),
            }
        )
        base_predictions_train.head(5)

        dt_train = dt_train.reshape(-1, 1)
        et_train = et_train.reshape(-1, 1)
        rf_train = rf_train.reshape(-1, 1)
        xg_train = xg_train.reshape(-1, 1)
        dt_test = dt_test.reshape(-1, 1)
        et_test = et_test.reshape(-1, 1)
        rf_test = rf_test.reshape(-1, 1)
        xg_test = xg_test.reshape(-1, 1)

        x_train = np.concatenate((dt_train, et_train, rf_train, xg_train), axis=1)
        x_test = np.concatenate((dt_test, et_test, rf_test, xg_test), axis=1)

        stk = xgb.XGBClassifier().fit(x_train, y_train)
        y_predict = stk.predict(x_test)
        y_true = y_test
        stk_score = accuracy_score(y_true, y_predict)
        print("Accuracy of Stacking: " + str(stk_score))
        precision, recall, fscore, none = precision_recall_fscore_support(
            y_true, y_predict, average="weighted"
        )
        f1 = f1_score(y_true, y_predict, average=None)
        print("Precision of Stacking: " + (str(precision)))
        print("Recall of Stacking: " + (str(recall)))
        print("F1-score of Stacking: " + (str(fscore)))
        print(classification_report(y_true, y_predict))
        cm = confusion_matrix(y_true, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        # plt.show()

        end_time = datetime.datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        results = {
            "model": self.name,
            "dataset": self.dataset,
            "parameters": self.parameters,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "total_duration": total_duration,
            "results": {
                "accuracy": stk_score,
                "precision": precision,
                "recall": recall,
                "avg_f1": fscore,
                "categorical_f1": f1.tolist(),
            },
        }

        return results, None
