import React from 'react';
import './MathDeepDive.css';

const MatrixDisplay = ({ title, data, rowHeaders, colHeaders, format = (v) => v.toExponential(3) }) => {
    if (!data || data.length === 0) return null;

    return (
        <div className="math-matrix-block">
            <h4>{title}</h4>
            <div className="math-table-container">
                <table className="math-table">
                    <thead>
                        <tr>
                            <th>t</th>
                            {colHeaders.map((c, i) => <th key={i}>{c}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((row, r) => (
                            <tr key={r}>
                                <th className="t-row-header">{r + 1} ({rowHeaders[r] || ''})</th>
                                {row.map((cell, c) => (
                                    <td key={c}>{format(cell)}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

const MathDeepDive = ({ stepData, O, states }) => {
    if (!stepData) {
        return (
            <div className="deep-dive-container glass-panel">
                <h3 className="section-title">Algorithmic Deep Dive</h3>
                <p className="empty-math">Run an iteration to see intermediate computations.</p>
            </div>
        );
    }

    const { alpha, beta, gamma } = stepData;

    return (
        <div className="deep-dive-container glass-panel">
            <h3 className="section-title">Algorithmic Deep Dive</h3>
            <p className="deep-dive-desc">
                Explore the exact intermediate probabilities computed during the E-step
                (Forward-Backward algorithm). Time steps $t$ correspond to the observations.
            </p>

            <div className="math-grid">
                <MatrixDisplay
                    title="Forward Probabilities (α)"
                    data={alpha}
                    rowHeaders={O}
                    colHeaders={states}
                />
                <MatrixDisplay
                    title="Backward Probabilities (β)"
                    data={beta}
                    rowHeaders={O}
                    colHeaders={states}
                />
                <MatrixDisplay
                    title="State Responsibilities (γ)"
                    data={gamma}
                    rowHeaders={O}
                    colHeaders={states}
                    format={(v) => v.toFixed(4)}
                />
            </div>
        </div>
    );
};

export default MathDeepDive;
