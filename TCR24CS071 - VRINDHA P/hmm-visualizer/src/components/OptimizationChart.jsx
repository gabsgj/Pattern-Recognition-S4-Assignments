import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts';
import './OptimizationChart.css';

const OptimizationChart = ({ history }) => {
    // Map history of p values to chart data format
    // We use log likelihood for better visualization, assuming p > 0
    const data = history.map((p, index) => ({
        iteration: index,
        likelihood: p,
        logLikelihood: p > 0 ? Math.log(p) : 0
    }));

    if (history.length === 0) {
        return (
            <div className="chart-container glass-panel">
                <h3 className="section-title">Convergence</h3>
                <div className="chart-empty">Run the algorithm to see optimization progress.</div>
            </div>
        );
    }

    return (
        <div className="chart-container glass-panel">
            <h3 className="section-title">Convergence (Log Likelihood)</h3>
            <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis
                            dataKey="iteration"
                            stroke="#a1a1aa"
                            tick={{ fill: '#a1a1aa', fontSize: 12, fontFamily: 'Inter' }}
                            tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                        />
                        <YAxis
                            domain={['auto', 'auto']}
                            stroke="#a1a1aa"
                            tickFormatter={(val) => val.toFixed(2)}
                            tick={{ fill: '#a1a1aa', fontSize: 12, fontFamily: 'Inter' }}
                            tickLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                                border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: '8px',
                                color: '#fff',
                                fontFamily: 'Inter',
                                boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
                            }}
                            formatter={(value, name) => [value.toFixed(4), name === 'logLikelihood' ? 'Log Likelihood' : 'Likelihood']}
                            labelFormatter={(label) => `Iteration ${label}`}
                        />
                        <Line
                            type="monotone"
                            dataKey="logLikelihood"
                            stroke="#ec4899"
                            strokeWidth={3}
                            dot={{ r: 4, fill: '#ec4899', strokeWidth: 2, stroke: '#1e293b' }}
                            activeDot={{ r: 6, fill: '#60a5fa', stroke: '#1e293b' }}
                            animationDuration={1500}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default OptimizationChart;
