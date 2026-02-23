import React from 'react';
import './BaumWelchVisualizer.css';

const BaumWelchVisualizer = ({ stepData, iterationCount, states, obsMap, O, onNextStep, onReset }) => {
    if (!stepData) {
        return (
            <div className="visualizer-container glass-panel">
                <h2 className="visualizer-title">Algorithm Visualization</h2>
                <p className="empty-state">Enter an observation sequence and press "Run First Step" to begin.</p>
                <button className="primary-button pulse" onClick={onNextStep}>Run First Step</button>
            </div>
        );
    }

    const { p, error, Pi, A, B } = stepData;

    return (
        <div className="visualizer-container glass-panel">
            <div className="visualizer-header">
                <h2 className="visualizer-title">Iteration {iterationCount}</h2>
                <div className="metrics">
                    <div className="metric-badge">
                        <span className="metric-label">P(O | λ)</span>
                        <span className="metric-value">{p ? p.toExponential(4) : '0'}</span>
                    </div>
                </div>
            </div>

            {error && <div className="error-alert">{error}</div>}

            <div className="visualization-grid">
                <div className="vis-section">
                    <h3>Updated Initial Probabilities (π)</h3>
                    <div className="bar-chart">
                        {Pi.map((val, i) => (
                            <div key={i} className="bar-container">
                                <div className="bar-label">{states[i]}</div>
                                <div className="bar-track">
                                    <div className="bar-fill" style={{ width: `${val * 100}%` }}></div>
                                </div>
                                <div className="bar-value">{val.toFixed(3)}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="controls">
                <button className="primary-button" onClick={onNextStep}>
                    Run Next Iteration
                </button>
                <button className="secondary-button" onClick={onReset}>
                    Reset Model
                </button>
            </div>
        </div>
    );
};

export default BaumWelchVisualizer;
