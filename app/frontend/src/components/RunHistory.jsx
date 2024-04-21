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

function RunHistory() {
  const [runHistory, setRunHistory] = useState([]);

  useEffect(() => {
    axios.get("/history").then((response) => {
      setRunHistory(response.data);
    });
  }, []);

  return (
    <div>
      <h1>Run History</h1>
    </div>
  );
}

export default RunHistory;