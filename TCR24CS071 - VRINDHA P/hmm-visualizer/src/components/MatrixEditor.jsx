import React from 'react';
import './MatrixEditor.css'; // We'll add some specific structural styles here if needed, but styling mostly in index.css

const MatrixEditor = ({ title, data, rowHeaders, colHeaders, onChange }) => {
    const handleChange = (r, c, val) => {
        let parsed = parseFloat(val);
        if (isNaN(parsed)) parsed = 0;

        // Copy matrix
        const newData = data.map(row => [...row]);
        newData[r][c] = parsed;
        onChange(newData);
    };

    return (
        <div className="matrix-editor glass-panel">
            <h3 className="matrix-title">{title}</h3>
            <table className="matrix-table">
                <thead>
                    <tr>
                        <th></th>
                        {colHeaders.map((c, i) => <th key={i}>{c}</th>)}
                    </tr>
                </thead>
                <tbody>
                    {data.map((row, r) => (
                        <tr key={r}>
                            <th>{rowHeaders[r] || ''}</th>
                            {row.map((cell, c) => (
                                <td key={c}>
                                    <input
                                        type="number"
                                        step="0.01"
                                        min="0"
                                        max="1"
                                        value={parseFloat(cell.toFixed(3))}
                                        onChange={(e) => handleChange(r, c, e.target.value)}
                                        className="matrix-input"
                                    />
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default MatrixEditor;
