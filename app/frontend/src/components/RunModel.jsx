import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Section,
  SectionCard,
  FormGroup,
  InputGroup,
  Button,
  HTMLSelect,
  Label,
  Icon,
  Popover,
  Menu,
  MenuItem,
  Card,
} from "@blueprintjs/core";
import RunResults from "./RunResults";
import { formatPythonVarName } from "../util";
import { showToast } from "../util/toaster";

//  results = {
//    "model": self.name,
//    "parameters": self.parameters,
//    "start_time": str(start_time),
//    "end_time": str(end_time),
//    "total_duration": total_duration,
//    "results": {
//    "accuracy": accuracy,
//    "precision": precision,
//    "recall": recall,
//    "avg_f1": avg_f1,
//    "categorical_f1": f1.tolist(),
//    },
//  }

const USE_MOCK_DATA = false;
const MOCK_MODEL_OUTPUT = {
  'model': 'LCCDE',
  'dataset': 'CICIDS2017_sample.csv',
  'parameters': {
    'lightgbm_classifier': {},
    'xgboost_classifier': {},
    'catboost_classifier': { 'boosting_type': 'Plain' }
  },
  'start_time': '2024-04-21 16:59:37.212374',
  'end_time': '2024-04-21 17:04:09.618285',
  'total_duration': 272.405911,
  'results': {
    'accuracy': 0.9975746268656717,
    'precision': 0.9975824670449086,
    'recall': 0.9975746268656717,
    'avg_f1': 0.9975490695240433,
    'categorical_f1': [0.9983597594313832, 0.9935316946959897, 1.0, 0.9983633387888707, 0.8571428571428571, 0.9935483870967742, 0.9977827050997783]
  }
};

const ADD_PARAM_BUTTON_STYLE = {
  alignItems: "left",
  margin: "10px",
  width: "200px",
  padding: "0",
  height: "25px",
};

function RunModel() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState({});
  const [parameters, setParameters] = useState({});
  const [customizedParameters, setCustomizedParameters] = useState({});
  const [modelOutput, setModelOutput] = useState();
  const [modelCurrentlyRunning, setModelCurrentlyRunning] = useState(false);
  const [runError, setRunError] = useState(null);
  const [dataset, setDataset] = useState("CICIDS2017_sample.csv");

  useEffect(() => {
    axios
      .get("/models")
      .then((response) => {
        setModels(response.data.models);
        if (response.data.models.length > 0) {
          setSelectedModel(response.data.models[0]);
          setParameters(response.data.models[0].parameters);
        }
      })
      .catch((error) => {
        const errorMsg = error.response.data.message ?? `${error}`;
        const output = `Error fetching models: ${errorMsg}`;
        console.error(output);
        showToast({ message: output, intent: "danger" })
      });
  }, []);

  const handleModelChange = (event) => {
    const model = models.find((model) => model.name === event.target.value);
    setSelectedModel(model);
    setParameters(model.parameters);
    setCustomizedParameters({});
  };

  const handleDatasetChange = (event) => {
    setDataset(event.target.value);
  };

  const handleAddCustomizedParameter = (event, submodelName, paramName) => {
    setCustomizedParameters((prevCustomizedParameters) => ({
      ...prevCustomizedParameters,
      [submodelName]: {
        ...prevCustomizedParameters[submodelName],
        [paramName]: "",
      },
    }));

    console.log(customizedParameters);
  };

  const handleCustomizedParameterChange = (
    event,
    submodelName,
    paramName
  ) => {
    const value = +event.target.value
      ? +event.target.value
      : event.target.value;
    setCustomizedParameters((prevCustomizedParameters) => ({
      ...prevCustomizedParameters,
      [submodelName]: {
        ...prevCustomizedParameters[submodelName],
        [paramName]: value,
      },
    }));
  };

  const handleClickRun = () => {
    // remove any parameters that have empty string as value
    const filteredParameters = {};
    for (const submodelName in customizedParameters) {
      filteredParameters[submodelName] = {};
      for (const paramName in customizedParameters[submodelName]) {
        if (customizedParameters[submodelName][paramName] !== "") {
          filteredParameters[submodelName][paramName] =
            customizedParameters[submodelName][paramName];
        }
      }
    }

    console.log("Running model with parameters:", filteredParameters);
    setRunError(null);
    setModelOutput(null);
    setModelCurrentlyRunning(true);
    if (USE_MOCK_DATA) {
      setTimeout(() => {
        setModelOutput({ ...MOCK_MODEL_OUTPUT, "parameters": filteredParameters });
        setModelCurrentlyRunning(false);
      }, 2000);
      return;
    }
    // send request to API
    axios
      .post("/run_engine", {
        model: selectedModel.name,
        parameters: filteredParameters,
        dataset: dataset,
      })
      .then((response) => {
        console.log("API response:", response.data);
        setModelOutput(response.data);
        setModelCurrentlyRunning(false);
      })
      .catch((error) => {
        const errorMsg = error.response.data.message ?? `${error}`;
        const output = `Error running model: ${errorMsg}`;
        console.error(output);
        setRunError(errorMsg);
        showToast({ message: output ?? `${error}`, intent: "danger" })
        setModelCurrentlyRunning(false);
      });
  };

  const getParameterLabelInfo = (submodelName, paramName) => {
    const paramObj = selectedModel.parameters[submodelName][paramName];
    if (paramObj.choices) {
      return "Choices: " + paramObj.choices.join(", ");
    } else if (paramObj.range) {
      return "Range: " + paramObj.range;
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        justifyContent: "space-around",
        alignItems: "flex-start",
        width: "100%",
        height: "75%",
        padding: "0 10%",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <h2>Select Model</h2>
        <HTMLSelect
          onChange={handleModelChange}
          iconName="caret-down"
          style={{
            width: "400px",
            height: "40px",
            color: "green",
            fontWeight: "bold",
            textAlign: "center",
            marginBottom: "20px",
            padding: "0px 30px",
          }}
        >
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name}
            </option>
          ))}
        </HTMLSelect>

        <h3>Select Dataset</h3>
        <HTMLSelect
          onChange={handleDatasetChange}
          iconName="caret-down"
          style={{
            width: "400px",
            height: "40px",
            color: "#555",
            fontWeight: "bold",
            textAlign: "center",
            marginBottom: "20px",
            padding: "0px 30px",
          }}
        >
          {["CICIDS2017_sample.csv", "CICIDS2017_sample_km.csv"].map((dataset) => (
            <option key={dataset} value={dataset}>
              {dataset}
            </option>
          ))}
        </HTMLSelect>

        <h3>Customize Parameters</h3>
        {selectedModel.parameters &&
          Object.keys(selectedModel.parameters).map(
            (submodelName) => (
              <Section
                title={formatPythonVarName(submodelName)}
                style={{
                  width: "400px",
                  padding: "0 20px",
                  margin: "10px",
                  border: "1px solid lightgrey",
                  borderRadius: "10px",
                  boxShadow: "0px 0px 10px lightgrey",
                }}
              >
                <Popover
                  position="right"
                  content={
                    <Menu>
                      {Object.keys(
                        selectedModel.parameters[
                        submodelName
                        ]
                      ).map((paramName) => (
                        <MenuItem
                          key={paramName}
                          text={paramName}
                          onClick={(event) =>
                            handleAddCustomizedParameter(
                              event,
                              submodelName,
                              paramName
                            )
                          }
                        />
                      ))}
                    </Menu>
                  }
                >
                  <Button
                    text="Add new parameter"
                    icon="add"
                    style={ADD_PARAM_BUTTON_STYLE}
                  />
                </Popover>
                {customizedParameters &&
                  customizedParameters[submodelName] &&
                  Object.keys(
                    customizedParameters?.[submodelName]
                  ).map((paramName) => (
                    <FormGroup
                      label={paramName}
                      labelFor={`${paramName}-text-input`}
                      labelInfo={getParameterLabelInfo(
                        submodelName,
                        paramName
                      )}
                      helperText={
                        selectedModel.parameters[
                          submodelName
                        ][paramName].description
                      }
                    >
                      <InputGroup
                        id={`${paramName}-text-input`}
                        type={
                          selectedModel.parameters[
                            submodelName
                          ][paramName].dtype ===
                            "int" ||
                            selectedModel.parameters[
                              submodelName
                            ][paramName].dtype ===
                            "float"
                            ? "number"
                            : "text"
                        }
                        step={
                          selectedModel.parameters[
                            submodelName
                          ][paramName].dtype === "int"
                            ? 1
                            : selectedModel
                              .parameters[
                              submodelName
                            ][paramName].dtype ===
                              "float"
                              ? 0.01
                              : undefined
                        }
                        placeholder={
                          selectedModel.parameters[
                            submodelName
                          ][paramName]
                            .model_default ??
                          selectedModel.parameters[
                            submodelName
                          ][paramName].default ??
                          ""
                        }
                        onChange={(event) =>
                          handleCustomizedParameterChange(
                            event,
                            submodelName,
                            paramName
                          )
                        }
                      />
                    </FormGroup>
                  ))}
              </Section>
            )
          )}
        <button
          onClick={handleClickRun}
          disabled={modelCurrentlyRunning}
          style={{
            marginTop: "20px",
            marginBottom: "60px",
            width: "200px",
            height: "40px",
            backgroundColor: "green",
            color: "white",
            fontWeight: "bold",
            textAlign: "center",
          }}
        >
          {modelCurrentlyRunning ? "Running..." : "Run Model"}
        </button>
      </div>
      <RunResults modelOutput={modelOutput} modelCurrentlyRunning={modelCurrentlyRunning} runError={runError} renderResultsTitle={true} />
    </div>
  );
}

export default RunModel;
