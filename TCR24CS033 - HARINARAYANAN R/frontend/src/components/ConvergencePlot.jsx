import React from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const ConvergencePlot = ({ logLikelihoods }) => {
    if (!logLikelihoods || logLikelihoods.length === 0) return null;

    const data = {
        labels: logLikelihoods.map((_, i) => i + 1),
        datasets: [
            {
                label: 'Log Likelihood',
                data: logLikelihoods,
                borderColor: '#a855f7',
                backgroundColor: 'rgba(168, 85, 247, 0.2)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#6366f1',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#6366f1',
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#cbd5e1',
                    font: {
                        family: "'Inter', sans-serif"
                    }
                }
            },
            title: {
                display: false,
            },
        },
        scales: {
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                },
                ticks: {
                    color: '#94a3b8',
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)',
                },
                ticks: {
                    color: '#94a3b8',
                },
                title: {
                    display: true,
                    text: 'Iteration',
                    color: '#cbd5e1'
                }
            },
        },
    };

    return (
        <div className="card shadow-glass plot-card">
            <h3 className="title-gradient">Training Convergence</h3>
            <div className="plot-container">
                <Line options={options} data={data} />
            </div>
        </div>
    );
};

export default ConvergencePlot;
