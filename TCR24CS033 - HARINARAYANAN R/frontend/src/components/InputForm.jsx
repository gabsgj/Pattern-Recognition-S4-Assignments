import React, { useState } from 'react';

const InputForm = ({ onTrain, isLoading }) => {
    const [observations, setObservations] = useState("A B A C B A");
    const [numStates, setNumStates] = useState(2);
    const [maxIter, setMaxIter] = useState(50);

    const handleSubmit = (e) => {
        e.preventDefault();
        const obsArray = observations.trim().split(/\s+/);
        if (obsArray.length === 0 || obsArray[0] === "") {
            alert("Please provide at least one observation.");
            return;
        }
        onTrain({ observations: obsArray, numStates, maxIter });
    };

    return (
        <div className="card input-card shadow-glass">
            <h2 className="title-gradient">Configure HMM Model</h2>
            <form onSubmit={handleSubmit} className="form-layout">
                <div className="form-group">
                    <label>Observation Sequence (space separated)</label>
                    <input
                        type="text"
                        value={observations}
                        onChange={(e) => setObservations(e.target.value)}
                        placeholder="e.g. A B C A B"
                        className="input-field"
                    />
                </div>

                <div className="form-row">
                    <div className="form-group">
                        <label>Hidden States</label>
                        <input
                            type="number"
                            min="2"
                            max="10"
                            value={numStates}
                            onChange={(e) => setNumStates(parseInt(e.target.value))}
                            className="input-field"
                        />
                    </div>
                    <div className="form-group">
                        <label>Max Iterations</label>
                        <input
                            type="number"
                            min="1"
                            max="500"
                            value={maxIter}
                            onChange={(e) => setMaxIter(parseInt(e.target.value))}
                            className="input-field"
                        />
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="btn-primary"
                >
                    {isLoading ? (
                        <span className="flex-center">
                            <span className="spinner"></span>
                            Training Model...
                        </span>
                    ) : (
                        'Train HMM'
                    )}
                </button>
            </form>
        </div>
    );
};

export default InputForm;
