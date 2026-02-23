import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MatrixEditor from './components/MatrixEditor';
import BaumWelchVisualizer from './components/BaumWelchVisualizer';
import Navigation from './components/Navigation';
import StateGraph from './components/StateGraph';
import OptimizationChart from './components/OptimizationChart';
import MathDeepDive from './components/MathDeepDive';
import { baumWelchStep } from './utils/hmm';
import './index.css';

const DEFAULT_STATES = ['Rainy', 'Sunny'];
const DEFAULT_OBS = ['Walk', 'Shop'];

const DEFAULT_PI = [0.6, 0.4];
const DEFAULT_A = [[0.7, 0.3], [0.4, 0.6]];
const DEFAULT_B = [[0.1, 0.9], [0.6, 0.4]];

const DEFAULT_SEQUENCE = 'Walk,Shop,Shop,Walk,Shop';

function App() {
  const [activeTab, setActiveTab] = useState('playground');

  const [states] = useState(DEFAULT_STATES);
  const [obsMap] = useState(DEFAULT_OBS);

  const [Pi, setPi] = useState(DEFAULT_PI);
  const [A, setA] = useState(DEFAULT_A);
  const [B, setB] = useState(DEFAULT_B);

  const [sequenceStr, setSequenceStr] = useState(DEFAULT_SEQUENCE);

  const [stepData, setStepData] = useState(null);
  const [iteration, setIteration] = useState(0);
  const [history, setHistory] = useState([]); // track p over time

  const handleNextStep = () => {
    const O_arr = sequenceStr.split(',').map(s => s.trim()).filter(Boolean);
    const currentPi = stepData ? stepData.Pi : Pi;
    const currentA = stepData ? stepData.A : A;
    const currentB = stepData ? stepData.B : B;

    const result = baumWelchStep(O_arr, obsMap, currentPi, currentA, currentB);

    setStepData(result);
    setIteration(prev => prev + 1);
    if (!result.error) {
      setHistory(prev => [...prev, result.p]);
    }
  };

  const handleReset = () => {
    setStepData(null);
    setIteration(0);
    setHistory([]);
    setPi(DEFAULT_PI);
    setA(DEFAULT_A);
    setB(DEFAULT_B);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'playground':
        return (
          <motion.div
            key="playground"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <div className="panels-grid">
              <div className="matrices-section">
                <MatrixEditor
                  title="Initial Probabilities (π)"
                  data={[stepData ? stepData.Pi : Pi]}
                  rowHeaders={['π']}
                  colHeaders={states}
                  onChange={(newData) => { if (!stepData) setPi(newData[0]); }}
                />
                <MatrixEditor
                  title="Transition Matrix (A)"
                  data={stepData ? stepData.A : A}
                  rowHeaders={states}
                  colHeaders={states}
                  onChange={(newData) => { if (!stepData) setA(newData); }}
                />
              </div>
              <div className="matrices-section">
                <MatrixEditor
                  title="Emission Matrix (B)"
                  data={stepData ? stepData.B : B}
                  rowHeaders={states}
                  colHeaders={obsMap}
                  onChange={(newData) => { if (!stepData) setB(newData); }}
                />
                <div className="sequence-input-container glass-panel">
                  <label className="sequence-label">Observation Sequence</label>
                  <input
                    type="text"
                    className="sequence-input"
                    value={sequenceStr}
                    onChange={(e) => setSequenceStr(e.target.value)}
                    disabled={!!stepData}
                  />
                  <span className="sequence-help">Comma-separated. Allowed: Walk, Shop</span>
                </div>
              </div>
            </div>

            <BaumWelchVisualizer
              stepData={stepData}
              iterationCount={iteration}
              states={states}
              obsMap={obsMap}
              O={sequenceStr.split(',').map(s => s.trim()).filter(Boolean)}
              onNextStep={handleNextStep}
              onReset={handleReset}
            />
          </motion.div>
        );
      case 'graph':
        return (
          <motion.div
            key="graph"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="full-height-tab"
          >
            <StateGraph states={states} A={stepData ? stepData.A : A} />
          </motion.div>
        );
      case 'optimization':
        return (
          <motion.div
            key="optimization"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="full-height-tab"
          >
            <OptimizationChart history={history} />
          </motion.div>
        );
      case 'deepdive':
        return (
          <motion.div
            key="deepdive"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <MathDeepDive
              stepData={stepData}
              states={states}
              O={sequenceStr.split(',').map(s => s.trim()).filter(Boolean)}
            />
          </motion.div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <motion.h1
          className="app-title"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, type: 'spring' }}
        >
          HMM Visualizer
        </motion.h1>
        <p className="app-description">
          A visual implementation of the Baum-Welch Algorithm.
        </p>
      </header>

      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />

      <div className="tab-content-area">
        <AnimatePresence mode="wait">
          {renderContent()}
        </AnimatePresence>
      </div>
    </div>
  );
}

export default App;
