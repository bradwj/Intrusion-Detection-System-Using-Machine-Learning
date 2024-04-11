import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { BrowserRouter as Router } from 'react-router-dom';

//function for changing the circle's color
function setCircleColor(color1, color2, circle) {
    document.querySelector('.circle').style.backgroundImage = `radial-gradient(${color1}, ${color2})`;
}

//function for resetting the circle's color
function resetCircleColor() {
    document.querySelector('.circle').style.backgroundImage = 'radial-gradient(orange, green)';
}

function RunTest() {
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState({});
    const [parameters, setParameters] = useState({});

    useEffect(() => {
        axios.get('/get_models')
            .then(response => {
                setModels(response.data.models);
                if (response.data.models.length > 0) {
                    setSelectedModel(response.data.models[0]);
                    setParameters(response.data.models[0].parameters);
                }
            })
            .catch(error => {
                console.error('Error fetching models:', error);
            });
    }, []);

    const handleModelChange = (event) => {
        const model = models.find(model => model.name === event.target.value);
        setSelectedModel(model);
        setParameters(model.parameters);
    };

    const handleParameterChange = (paramName, value) => {
        setParameters(prevParameters => ({
            ...prevParameters,
            [paramName]: value
        }));
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <h2 style={{ marginBottom: '10px' }}>Select Model:</h2>
            <select
                onChange={handleModelChange}
                style={{
                    width: '400px',
                    height: '40px',
                    backgroundColor: 'white',
                    color: 'green',
                    fontWeight: 'bold',
                    textAlign: 'center'
                }}
            >
                {models.map(model => (
                    <option key={model.name} value={model.name}>
                        {model.name}
                    </option>
                ))}
            </select>
            <h2 style={{ marginTop: '20px', marginBottom: '10px' }}>Parameters:</h2>
            {selectedModel.parameters && Object.keys(selectedModel.parameters).map(paramName => (
                <div key={paramName}>
                    <label>{paramName}:</label>
                    <input
                        type="text"
                        value={parameters[paramName]}
                        onChange={(e) => handleParameterChange(paramName, e.target.value)}
                    />
                </div>
            ))}
            <button
                style={{
                    marginTop: '20px',
                    width: '200px',
                    height: '40px',
                    backgroundColor: 'green',
                    color: 'white',
                    fontWeight: 'bold',
                    textAlign: 'center'
                }}
            >
                Run
            </button>
        </div>
    );
}

function App() {
    const [showDropdown, setShowDropdown] = useState(false);

    return (
        <Router>
            <div className="App">
                <header>
                    <h1>Welcome to Team 71's Development Tool for ML-based Models for IDS</h1>
                    <p>This tool was designed by Olivia Barlow, Bradley Johnson, Sasha Kaplan, and Gwen Logsdon in coordination with Dr. Nhut Nguyen and the University of Texas at Dallas.</p>
                </header>
                <div className="App-content">
                    <button className="green-gradient" onMouseEnter={() => setCircleColor('white', 'green')} onMouseLeave={resetCircleColor} onClick={() => setShowDropdown(true)}>
                        Run Test
                    </button>
                    <div className="circle"></div>
                    <button className="orange-gradient" onMouseEnter={() => setCircleColor('white', 'orange')} onMouseLeave={resetCircleColor} onClick={() => setShowDropdown(false)}>
                        Retrieve and Compare Data
                    </button>
                </div>
                {showDropdown && <RunTest />}
            </div>
        </Router>
    );
}

export default App;
