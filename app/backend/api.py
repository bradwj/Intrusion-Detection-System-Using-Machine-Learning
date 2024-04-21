from flask import Flask, request
from engine import *
from tinydb import TinyDB, Query
import logging
import sys

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.config["JSON_SORT_KEYS"] = False

db = TinyDB("db.json")


@app.route("/", methods=["GET"])
def hello_world():
    return "Hello, World!"


@app.route("/models", methods=["GET"])
def get_models():
    data = {"models": Model.get_models()}
    app.logger.info(data)
    return data, 200


@app.route("/history", methods=["GET"])
def get_history():
    history = db.all()
    return {"history": history}, 200


# Request body format:
# {
#     "model": $model_name // one of [LCCDE, MTH, TreeBased]
#     "parameters": {
#         $submodel_name_1: {
#             $parameter_name_1: $parameter_value_1,
#             $parameter_name_2: $parameter_value_2,
#         },
#         $submodel_name_2: {
#             $parameter_name_1: $parameter_value_1,
#             $parameter_name_2: $parameter_value_2,
#         },
#     }
# }
@app.route("/run_engine", methods=["POST"])
def run_engine():
    try:
        data = request.json
        if "model" not in data:
            return {"message": "'model' field is required"}, 400

        model_name = data["model"]
        parameters = data["parameters"] if "parameters" in data else {}
        model, err = Model.get_model(model_name)
        if err:
            return {"message": err}, 400

        app.logger.info(f"Setting parameters")
        if err := model.set_parameters(parameters):
            return {"message": err}, 400

        output, err = model.run()
        if err:
            return {"message": err}, 400

        app.logger.info(output)

        db.insert(output)
        return output, 200

    except Exception as e:
        app.logger.error(e)
        return {"message": "Internal server error"}, 500
