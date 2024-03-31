import React, { useState } from 'react';
import './App.css';
import { BrowserRouter as Router } from 'react-router-dom';

function setCircleColor(color1, color2, circle) {
    document.querySelector('.circle').style.backgroundImage = `radial-gradient(${color1}, ${color2})`;
}

function resetCircleColor() {
    document.querySelector('.circle').style.backgroundImage = 'radial-gradient(orange, green)';
}

function RunTest() {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <h2 style={{ marginBottom: '10px' }}>Select Model:</h2>
            <select style={{
                width: '400px',
                height: '40px',
                backgroundColor: 'white',
                color: 'green',
                fontWeight: 'bold',
                textAlign: 'center'
            }}>
                <option value="alg1">Tree-Based IDS</option>
                <option value="alg2">The Leader Class and Confidence Decision Ensemble</option>
                <option value="alg3">Multi-Tiered Hybrid IDS</option>
            </select>
            <h2 style={{ marginTop: '20px', marginBottom: '10px' }}>Select Dataset:</h2>
            <select style={{
                width: '400px',
                height: '40px',
                backgroundColor: 'white',
                color: 'green',
                fontWeight: 'bold',
                textAlign: 'center'
            }}>
                <option value="dataset1">Dataset 1</option>
                <option value="dataset2">Dataset 2</option>
                <option value="dataset3">Dataset 3</option>
            </select>
            <button style={{
                marginTop: '20px',
                width: '200px',
                height: '40px',
                backgroundColor: 'green',
                color: 'white',
                fontWeight: 'bold',
                textAlign: 'center'
            }}>
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
