import React, { useState, useEffect } from "react";
import axios from "axios";
import { Column, Cell, Table } from "@blueprintjs/table";
import { Button, HTMLTable } from "@blueprintjs/core";
import RunResults from "./RunResults";
import { formatPythonVarName } from "../util";
import { showToast } from "../util/toaster";

const MAX_SELECTED_ROWS = 3;

function RunHistory() {
  const [runHistory, setRunHistory] = useState([]);
  const [selectedModelIndexes, setSelectedIndexes] = useState([]);

  useEffect(() => {
    axios.get("/history")
      .then((response) => {
        console.log(response.data);
        setRunHistory(response.data.history);
      }).catch((error) => {
        console.error("Error fetching run history: ", error);
        showToast({ message: `Error fetching run history. Please try again. ${error}`, intent: "danger" })
      });
  }, []);

  /* the select button should select the row and console.log all of the data stored in runHistory for that model*/
  const handleClickSelectRow = (event, idx) => {
    console.log("Row selected: ", idx);
    if (selectedModelIndexes.includes(idx)) {
      setSelectedIndexes(prev => prev.filter((i) => i !== idx));
    }
    else if (selectedModelIndexes.length < MAX_SELECTED_ROWS) {
      setSelectedIndexes(prev => [...prev, idx].sort((a, b) => a - b));
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "row",
        justifyContent: "space-around",
        alignItems: "flex-start",
        width: "100%",
        height: "75%",
        padding: "0",
      }}
    >
      <div style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",

      }}>
        <h2>Run History</h2>
        <div style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-evenly",
          alignItems: "flex-start",
          padding: "0",
        }}>
          {selectedModelIndexes.length > 0 && selectedModelIndexes.map(idx => <RunResults modelOutput={runHistory[idx]} rowIndex={idx} key={idx} />)}
        </div>
        <div style={{
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-around",
          alignItems: "center",
          width: "100%",
          height: "10%",
          padding: "0 10%",
        }}>
          {runHistory.length > 0 &&
            <HTMLTable width="100%">
              <thead>
                <tr>
                  <th></th>
                  <th>Model</th>
                  <th>Run Time</th>
                  <th>Duration (seconds)</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {runHistory.map((run, idx) => (
                  <tr key={idx} style={{
                    backgroundColor: selectedModelIndexes.includes(idx) ? "#f7e3c8" : null,
                    cursor: "pointer",
                  }}>
                    <td>{idx}</td>
                    <td>{run.model}</td>
                    <td>{run.start_time}</td>
                    <td>{Math.round(run.total_duration)}</td>
                    <td>{run.results.accuracy.toFixed(4)}</td>
                    <td>
                      <Button
                        onClick={(event) => handleClickSelectRow(event, idx)}
                        text={selectedModelIndexes.includes(idx) ? "Deselect" : "Select"}
                        intent={selectedModelIndexes.includes(idx) ? "success" : "none"}
                        disabled={!selectedModelIndexes.includes(idx) && selectedModelIndexes.length >= MAX_SELECTED_ROWS}
                        style={{ height: "50%" }}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </HTMLTable>
          }
        </div>
      </div>
    </div>
  );
}

export default RunHistory;