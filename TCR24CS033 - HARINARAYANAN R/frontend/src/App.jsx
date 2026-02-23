import React, { useState } from 'react';
import axios from 'axios';
import { Activity } from 'lucide-react';

import InputForm from './components/InputForm';
import MatrixView from './components/MatrixView';
import GraphView from './components/GraphView';
import ConvergencePlot from './components/ConvergencePlot';

import './index.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [observationsMap, setObservationsMap] = useState([]);

  const handleTrain = async (config) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/train', {
        observations: config.observations,
        num_states: config.numStates,
        max_iter: config.maxIter
      });
      setModelData(response.data);
      setObservationsMap(response.data.observations_map || Array.from({ length: response.data.B[0].length }, (_, i) => `O${i}`));
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "Failed to train model. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header shadow-glass">
        <div className="logo-container">
          <Activity className="logo-icon" />
          <h1 className="logo-text">HMM Visualizer <span className="logo-highlight">Pro</span></h1>
        </div>
        <p className="subtitle">Baum-Welch Algorithm from Scratch</p>
      </header>

      <main className="main-content">
        {error && (
          <div className="error-banner">
            <p>{error}</p>
          </div>
        )}

        <div className="top-section">
          <div className="input-section">
            <InputForm onTrain={handleTrain} isLoading={loading} />
          </div>

          <div className="convergence-section">
            {modelData ? (
              <ConvergencePlot logLikelihoods={modelData.log_likelihood} />
            ) : (
              <div className="placeholder-card card shadow-glass">
                <div className="placeholder-content">
                  <Activity className="placeholder-icon pulse" />
                  <p>Train the model to see convergence</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {modelData && (
          <>
            <div className="matrices-section">
              <MatrixView
                title="Transition Matrix (A)"
                matrix={modelData.A}
                labels={Array.from({ length: modelData.A.length }, (_, i) => `S${i}`)}
                colLabelsTitle="State"
              />
              <MatrixView
                title="Emission Matrix (B)"
                matrix={modelData.B}
                labels={observationsMap}
                colLabelsTitle="Observation"
              />
            </div>

            <div className="graph-section">
              <GraphView transitionMatrix={modelData.A} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
