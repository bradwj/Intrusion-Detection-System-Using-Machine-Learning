import React, { useState } from 'react';
import './App.css';

function App() {
  const [circleColor, setCircleColor] = useState('blue'); // Initial color set to blue

  return (
    <div className="App">
      <header>
        <h1>Welcome to Team 71's Development Tool for ML-based Algorithms for IDS</h1>
        <p>This tool was designed by Olivia Barlow, Bradley Johnson, Sasha Kaplan, and Gwen Logsdon in coordination with Dr. Nhut Nguyen and the University of Texas at Dallas.</p>
      </header>
      <div className="circle" style={{ backgroundcolor: circleColor }}></div>
      <div>
        <button onMouseEnter={() => setCircleColor('#ff0000')} onMouseLeave={() => setCircleColor('white')}>Run Test</button>
        <button onMouseEnter={() => setCircleColor('#00ff00')} onMouseLeave={() => setCircleColor('white')}>Retrieve Data</button>
        <button onMouseEnter={() => setCircleColor('#0000ff')} onMouseLeave={() => setCircleColor('white')}>Compare Data</button>
      </div>
    </div>
  );
}

export default App;