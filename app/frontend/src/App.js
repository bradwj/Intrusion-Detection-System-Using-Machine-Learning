import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { BrowserRouter as Router } from 'react-router-dom';
import RunModel from './components/RunModel';
import { Button } from '@blueprintjs/core';

//function for changing the circle's color
function setCircleColor(color1, color2, circle) {
    document.querySelector('.circle').style.backgroundImage = `radial-gradient(${color1}, ${color2})`;
}

//function for resetting the circle's color
function resetCircleColor() {
    document.querySelector('.circle').style.backgroundImage = 'radial-gradient(orange, green)';
}

function App() {
    const [showRunModelDropdown, setShowRunModelDropdown] = useState(false);

    return (
        <Router>
            <div className="App">
                <header>
                    <h1>Welcome to Team 71's Development Tool for ML-based Models for IDS</h1>
                    <p>This tool was designed by Olivia Barlow, Bradley Johnson, Sasha Kaplan, and Gwen Logsdon in coordination with Dr. Nhut Nguyen and the University of Texas at Dallas.</p>
                </header>
                <div className="App-content">
                    <button className="green-gradient" onMouseEnter={() => setCircleColor('white', 'green')} onMouseLeave={resetCircleColor} onClick={() => setShowRunModelDropdown(true)}>
                        Run Model
                    </button>
                    <div className="circle"></div>
                    <button className="orange-gradient" onMouseEnter={() => setCircleColor('white', 'orange')} onMouseLeave={resetCircleColor} onClick={() => setShowRunModelDropdown(false)}>
                        Compare Runs
                    </button>
                </div>
                {showRunModelDropdown && <RunModel />}
            </div>
        </Router>
    );
}

export default App;
