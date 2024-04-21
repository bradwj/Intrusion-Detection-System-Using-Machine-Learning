import React, { useState } from "react";
import {
    Section,
    SectionCard,
    Card,
    CardList,
    Label,
    Text,
    Spinner,
    Button,
    Collapse,
    Pre,
} from "@blueprintjs/core";
import { formatPythonVarName } from "../util";

// result format:
//        results = {
//            "model": self.name,
//            "parameters": self.parameters,
//            "start_time": str(start_time),
//            "end_time": str(end_time),
//            "total_duration": total_duration,
//            "results": {
//                "accuracy": accuracy,
//                "precision": precision,
//                "recall": recall,
//                "avg_f1": avg_f1,
//                "categorical_f1": f1.tolist(),
//            },
//        }

const STYLE_METRIC_CARD = {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 30px",
    margin: "5px",
    width: "100%",
    height: "50%",
    border: "1px solid lightgrey",
    borderRadius: "10px",
    boxShadow: "0px 0px 10px lightgrey",
}

const STYLE_METRIC_LABEL = {
    fontSize: "18px",
    margin: "0"
}

const STYLE_METRIC_VALUE = {
    fontSize: "20px",
    margin: "0",
    fontWeight: "bold",
    padding: "1px 3px",
    border: "1px solid lightgrey",
    borderRadius: "5px",
    boxShadow: "0px 0px 10px lightgrey",
    maxWidth: "50%",

}

function RunResults({ modelOutput, modelCurrentlyRunning }) {
    const [showParameters, setShowParameters] = useState(false);

    const handleToggleShowParameters = () => {
        setShowParameters(value => !value);
    }

    return (
        <div
            style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                width: "400px",
                padding: "0 20px",
                margin: "10px",
            }}
        >
            <h2>Results</h2>
            {
                modelOutput &&
                Object.keys(modelOutput.results).map((metric, index) =>
                    <Card style={STYLE_METRIC_CARD} key={index}
                    >
                        <p style={STYLE_METRIC_LABEL}>{formatPythonVarName(metric)}</p>
                        {Array.isArray(modelOutput.results[metric]) ? (
                            <p style={{
                                ...STYLE_METRIC_VALUE,
                                fontSize: "16px",
                            }}>
                                {modelOutput.results[metric].map(value => value.toFixed(3)).join(", ")}
                            </p>
                        ) : (
                            <p style={{
                                ...STYLE_METRIC_VALUE,
                                backgroundColor: `rgba(${255 - (modelOutput.results[metric] * 255)}, ${(modelOutput.results[metric] * 255)}, 0, 0.5)`
                            }}>
                                {/* round metric to 4 decimal places */}
                                {modelOutput.results[metric].toFixed(6)}
                            </p>
                        )}
                    </Card>

                )
            }
            {!modelOutput && !modelCurrentlyRunning && <p> Run the model first to view results.</p>}
            {
                !modelOutput && modelCurrentlyRunning &&
                <>
                    <p> Model is currently running. Please wait for results.</p>
                    <Spinner size="100" intent="success" style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        width: "100%",
                        height: "100%",
                        padding: "10px"
                    }} />
                </>
            }

            {modelOutput && (
                <div
                    style={{
                        padding: "20px 10px",
                        textAlign: "left",
                        width: "100%",
                    }}>
                    <Text>Model: {modelOutput.model}</Text>
                    <Text>Start time: {modelOutput.start_time}</Text>
                    <Text>End time: {modelOutput.end_time}</Text>
                    <Text>Total Duration: {modelOutput.total_duration.toFixed(2)} seconds</Text>
                    <Button onClick={handleToggleShowParameters} fill={true} style={{ padding: "0", margin: "0", height: "10px" }}>
                        {showParameters ? "Hide" : "Show"} Parameters
                    </Button>
                    <Collapse isOpen={showParameters}>
                        <Pre>{JSON.stringify(modelOutput.parameters, null, 2)}</Pre>
                    </Collapse>
                </div>
            )}
        </div >
    );
}

export default RunResults;
