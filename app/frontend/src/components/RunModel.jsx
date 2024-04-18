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
} from "@blueprintjs/core";

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

    useEffect(() => {
        axios
            .get("/get_models")
            .then((response) => {
                setModels(response.data.models);
                if (response.data.models.length > 0) {
                    setSelectedModel(response.data.models[0]);
                    setParameters(response.data.models[0].parameters);
                }
            })
            .catch((error) => {
                console.error("Error fetching models:", error);
            });
    }, []);

    const handleModelChange = (event) => {
        const model = models.find((model) => model.name === event.target.value);
        setSelectedModel(model);
        setParameters(model.parameters);
        setCustomizedParameters({});
    };

    const formatSubmodelName = (submodelName) => {
        return submodelName
            .split("_")
            .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
            .join(" ");
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
        console.log("Running model with parameters:", customizedParameters);
        // send request to API
        axios
            .post("/run_engine", {
                model: selectedModel.name,
                parameters: customizedParameters,
            })
            .then((response) => {
                console.log("API response:", response.data);
            })
            .catch((error) => {
                console.error("Error running model:", error);
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
            <h2>Customize Parameters</h2>
            {selectedModel.parameters &&
                Object.keys(selectedModel.parameters).map((submodelName) => (
                    <Section
                        title={formatSubmodelName(submodelName)}
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
                                        selectedModel.parameters[submodelName]
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
                                        selectedModel.parameters[submodelName][
                                            paramName
                                        ].description
                                    }
                                >
                                    <InputGroup
                                        id={`${paramName}-text-input`}
                                        type={
                                            selectedModel.parameters[
                                                submodelName
                                            ][paramName].dtype === "int" ||
                                            selectedModel.parameters[
                                                submodelName
                                            ][paramName].dtype === "float"
                                                ? "number"
                                                : "text"
                                        }
                                        step={
                                            selectedModel.parameters[
                                                submodelName
                                            ][paramName].dtype === "int"
                                                ? 1
                                                : selectedModel.parameters[
                                                      submodelName
                                                  ][paramName].dtype === "float"
                                                ? 0.01
                                                : undefined
                                        }
                                        placeholder={
                                            selectedModel.parameters[
                                                submodelName
                                            ][paramName].model_default ??
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
                ))}
            <button
                onClick={handleClickRun}
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
                Run
            </button>
        </div>
    );
}

export default RunModel;
