import React from 'react';

const MatrixRow = ({ row, rowIndex, labels, title }) => (
    <div className="matrix-row">
        <div className="matrix-row-label">
            {title === "Emission" ? `S${rowIndex}` : `S${rowIndex}`}
        </div>
        <div className="matrix-cells">
            {row.map((val, colIndex) => {
                const intensity = Math.max(0, Math.min(1, val));
                const bgOpacity = 0.1 + intensity * 0.9;
                const color = title === "Transition" ? `rgba(99, 102, 241, ${bgOpacity})` : `rgba(168, 85, 247, ${bgOpacity})`;
                return (
                    <div
                        key={colIndex}
                        className="matrix-cell"
                        style={{ backgroundColor: color }}
                    >
                        <span className="cell-value">
                            {val.toFixed(3)}
                        </span>
                    </div>
                );
            })}
        </div>
    </div>
);

const MatrixView = ({ title, matrix, labels, colLabelsTitle }) => {
    if (!matrix || matrix.length === 0) return null;

    return (
        <div className="card matrix-card shadow-glass">
            <h3 className="title-gradient">{title} Matrix</h3>
            <div className="matrix-container">
                <div className="matrix-header">
                    <div className="matrix-corner"></div>
                    <div className="matrix-col-labels">
                        {labels.map((label, i) => (
                            <div key={i} className="matrix-col-label">
                                {colLabelsTitle === "State" ? `S${i}` : label}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="matrix-body">
                    {matrix.map((row, i) => (
                        <MatrixRow key={i} row={row} rowIndex={i} labels={labels} title={title.split(" ")[0]} />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default MatrixView;
