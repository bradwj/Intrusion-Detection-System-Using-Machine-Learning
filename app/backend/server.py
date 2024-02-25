from flask import Flask, request
from engine import *
import logging
import sys

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

ENGINE = MLEngine()


@app.route("/", methods=["GET"])
def hello_world():
    return "Hello, World!"


@app.route("/load_data", methods=["GET"])
def load_data():
    ENGINE.load_dataset()
    return "Loaded dataset"


@app.route("/run_engine", methods=["POST"])
def run_engine():
    data = request.json
    if data["model"] not in EngineModels._member_names_:
        return (
            f"Model '{data['model']}' is not a valid model. Valid models: {EngineModels._member_names_}",
            400,
        )
    app.logger.info(f"Running model: {data['model']}")
    results = ENGINE.run_engine(data["model"])
    return results, 200
