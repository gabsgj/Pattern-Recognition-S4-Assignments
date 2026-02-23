// DOM Elements
const statesInput = document.getElementById('statesInput');
const obsInput = document.getElementById('obsInput');
const updateModelBtn = document.getElementById('updateModelBtn');
const matricesSection = document.getElementById('matricesSection');
const piMatrix = document.getElementById('piMatrix');
const aMatrix = document.getElementById('aMatrix');
const bMatrix = document.getElementById('bMatrix');
const sequenceInput = document.getElementById('sequenceInput');
const runIterBtn = document.getElementById('runIterBtn');
const runMultiBtn = document.getElementById('runMultiBtn');
const runConvBtn = document.getElementById('runConvBtn');
const resetBtn = document.getElementById('resetBtn');
const resultsSection = document.getElementById('resultsSection');
const iterationLog = document.getElementById('iterationLog');
const metricsPanel = document.getElementById('metricsPanel');
const valIteration = document.getElementById('valIteration');
const valLogLikelihood = document.getElementById('valLogLikelihood');
const valDelta = document.getElementById('valDelta');
const valStatus = document.getElementById('valStatus');

// State Variables
let states = [];
let observations = [];
let pi = [];
let A = [];
let B = [];
let iterationCount = 0;
let gammaChartInstance = null;
let llChartInstance = null;
let networkInstance = null;
let llHistory = [];
let lastLogLikelihood = null;
let statusText = "Initialized";

// Default values as per typical HMM example
const defaultPi = [0.6, 0.4];
const defaultA = [[0.7, 0.3], [0.4, 0.6]];
const defaultB = [[0.1, 0.9], [0.6, 0.4]];

updateModelBtn.addEventListener('click', () => {
    states = statesInput.value.split(',').map(s => s.trim()).filter(s => s);
    observations = obsInput.value.split(',').map(s => s.trim()).filter(s => s);

    if (states.length === 0 || observations.length === 0) {
        alert("Please provide valid lists of states and observations.");
        return;
    }

    renderMatrices();
    matricesSection.style.display = 'block';
});

function renderMatrices() {
    piMatrix.style.gridTemplateColumns = `repeat(${states.length}, 1fr)`;
    aMatrix.style.gridTemplateColumns = `repeat(${states.length}, 1fr)`;
    bMatrix.style.gridTemplateColumns = `repeat(${observations.length}, 1fr)`;

    piMatrix.innerHTML = '';
    aMatrix.innerHTML = '';
    bMatrix.innerHTML = '';

    // Initialize/Render Pi
    pi = [];
    states.forEach((state, i) => {
        const val = (states.length === 2) ? defaultPi[i] : (1 / states.length).toFixed(4);
        piMatrix.innerHTML += `
            <div class="matrix-cell">
                <span>&#960;(${state})</span>
                <input type="number" step="0.01" min="0" max="1" id="pi_${i}" value="${val}">
            </div>`;
    });

    // Initialize/Render A
    A = [];
    states.forEach((stateFrom, i) => {
        states.forEach((stateTo, j) => {
            const val = (states.length === 2) ? defaultA[i][j] : (1 / states.length).toFixed(4);
            aMatrix.innerHTML += `
                <div class="matrix-cell">
                    <span>A(${stateFrom}&rarr;${stateTo})</span>
                    <input type="number" step="0.01" min="0" max="1" id="a_${i}_${j}" value="${val}">
                </div>`;
        });
    });

    // Initialize/Render B
    B = [];
    states.forEach((state, i) => {
        observations.forEach((obs, j) => {
            const val = (states.length === 2 && observations.length === 2) ? defaultB[i][j] : (1 / observations.length).toFixed(4);
            bMatrix.innerHTML += `
                <div class="matrix-cell">
                    <span>B(${state}&rarr;${obs})</span>
                    <input type="number" step="0.01" min="0" max="1" id="b_${i}_${j}" value="${val}">
                </div>`;
        });
    });
}

function readMatrices() {
    const N = states.length;
    const M = observations.length;

    let currentPi = [];
    for (let i = 0; i < N; i++) {
        currentPi.push(parseFloat(document.getElementById(`pi_${i}`).value));
    }

    let currentA = [];
    for (let i = 0; i < N; i++) {
        let row = [];
        for (let j = 0; j < N; j++) {
            row.push(parseFloat(document.getElementById(`a_${i}_${j}`).value));
        }
        currentA.push(row);
    }

    let currentB = [];
    for (let i = 0; i < N; i++) {
        let row = [];
        for (let j = 0; j < M; j++) {
            row.push(parseFloat(document.getElementById(`b_${i}_${j}`).value));
        }
        currentB.push(row);
    }

    return { currentPi, currentA, currentB };
}

function updateDOMMatrices(newPi, newA, newB) {
    const N = states.length;
    const M = observations.length;

    for (let i = 0; i < N; i++) {
        document.getElementById(`pi_${i}`).value = newPi[i].toFixed(4);
    }

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            document.getElementById(`a_${i}_${j}`).value = newA[i][j].toFixed(4);
        }
    }

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) {
            document.getElementById(`b_${i}_${j}`).value = newB[i][j].toFixed(4);
        }
    }
}

function runIteration() {
    const seqStr = sequenceInput.value.split(',').map(s => s.trim()).filter(s => s);
    const O = seqStr.map(s => observations.indexOf(s));

    if (O.includes(-1)) {
        alert("Observation sequence contains an invalid observation symbol!");
        return;
    }

    const { currentPi, currentA, currentB } = readMatrices();
    const T = O.length;
    const N = states.length;

    // Scaling factors array
    let c = Array(T).fill(0);

    // 1. Forward Pass (Alpha) with scaling
    let alpha = Array(T).fill(0).map(() => Array(N).fill(0));

    for (let i = 0; i < N; i++) {
        alpha[0][i] = currentPi[i] * currentB[i][O[0]];
        c[0] += alpha[0][i];
    }

    // Scale alpha[0]
    c[0] = (c[0] === 0) ? 1e-300 : 1 / c[0];
    for (let i = 0; i < N; i++) {
        alpha[0][i] *= c[0];
    }

    for (let t = 1; t < T; t++) {
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let i = 0; i < N; i++) {
                sum += alpha[t - 1][i] * currentA[i][j];
            }
            alpha[t][j] = sum * currentB[j][O[t]];
            c[t] += alpha[t][j];
        }

        // Scale alpha[t]
        c[t] = (c[t] === 0) ? 1e-300 : 1 / c[t];
        for (let j = 0; j < N; j++) {
            alpha[t][j] *= c[t];
        }
    }

    // 2. Backward Pass (Beta) with scaling
    let beta = Array(T).fill(0).map(() => Array(N).fill(0));
    for (let i = 0; i < N; i++) {
        beta[T - 1][i] = 1 * c[T - 1];
    }

    for (let t = T - 2; t >= 0; t--) {
        for (let i = 0; i < N; i++) {
            let sum = 0;
            for (let j = 0; j < N; j++) {
                sum += currentA[i][j] * currentB[j][O[t + 1]] * beta[t + 1][j];
            }
            beta[t][i] = sum * c[t];
        }
    }

    // Log-Likelihood P(O|lambda)
    let logLikelihood = 0;
    for (let t = 0; t < T; t++) {
        logLikelihood -= Math.log(c[t]);
    }

    let delta = 0;
    if (lastLogLikelihood !== null) {
        delta = logLikelihood - lastLogLikelihood;
        if (Math.abs(delta) < 1e-4) {
            statusText = "Converged";
        } else {
            statusText = "Iterating";
        }
    } else {
        statusText = "Started";
    }

    lastLogLikelihood = logLikelihood;
    llHistory.push(logLikelihood);

    // 3. State Responsibilities (Gamma) and Transitions (Xi)
    // Because of scaling, Gamma and Xi formulas are simplified:
    let gamma = Array(T).fill(0).map(() => Array(N).fill(0));
    let xi = Array(T - 1).fill(0).map(() => Array(N).fill(0).map(() => Array(N).fill(0)));

    for (let t = 0; t < T; t++) {
        for (let i = 0; i < N; i++) {
            let denom = 0;
            for (let j = 0; j < N; j++) denom += alpha[t][j] * beta[t][j] / c[t];
            denom = denom === 0 ? 1 : denom;
            gamma[t][i] = (alpha[t][i] * beta[t][i] / c[t]) / denom;
        }

        if (t < T - 1) {
            let denomXi = 0;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    denomXi += alpha[t][i] * currentA[i][j] * currentB[j][O[t + 1]] * beta[t + 1][j];
                }
            }
            denomXi = denomXi === 0 ? 1 : denomXi;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    xi[t][i][j] = (alpha[t][i] * currentA[i][j] * currentB[j][O[t + 1]] * beta[t + 1][j]) / denomXi;
                }
            }
        }
    }

    // 4. Updates (Baum-Welch)
    let newPi = [...gamma[0]];

    let newA = Array(N).fill(0).map(() => Array(N).fill(0));
    for (let i = 0; i < N; i++) {
        let denomA = 0;
        for (let t = 0; t < T - 1; t++) denomA += gamma[t][i];

        for (let j = 0; j < N; j++) {
            let numA = 0;
            for (let t = 0; t < T - 1; t++) numA += xi[t][i][j];
            newA[i][j] = (denomA === 0) ? 0 : numA / denomA;
        }
    }

    let newB = Array(N).fill(0).map(() => Array(observations.length).fill(0));
    for (let i = 0; i < N; i++) {
        let denomB = 0;
        for (let t = 0; t < T; t++) denomB += gamma[t][i];

        for (let k = 0; k < observations.length; k++) {
            let numB = 0;
            for (let t = 0; t < T; t++) {
                if (O[t] === k) numB += gamma[t][i];
            }
            newB[i][k] = (denomB === 0) ? 0 : numB / denomB;
        }
    }

    iterationCount++;
    displayResults(alpha, beta, gamma, newPi, newA, newB);
    updateDOMMatrices(newPi, newA, newB);
    updateChart(gamma);
    updateLLChart();
    updateNetwork(newA);
    updateMetrics(logLikelihood, delta);
    return Math.abs(delta);
}

function displayResults(alpha, beta, gamma, pi_new, a_new, b_new) {
    resultsSection.style.display = 'flex';

    let html = `
        <div class="iteration-card">
            <h3>Iteration ${iterationCount}</h3>
            
            <div class="table-wrapper">
                <h4>Forward Probabilities (&#945;)</h4>
                <table>
                    <tr><th>Time Step</th>${states.map(s => `<th>${s}</th>`).join('')}</tr>
                    ${alpha.map((row, t) => `<tr><td>t=${t + 1} (${sequenceInput.value.split(',')[t].trim()})</td>${row.map(val => `<td>${val.toExponential(4)}</td>`).join('')}</tr>`).join('')}
                </table>
            </div>

            <div class="table-wrapper">
                <h4>Backward Probabilities (&#946;)</h4>
                <table>
                    <tr><th>Time Step</th>${states.map(s => `<th>${s}</th>`).join('')}</tr>
                    ${beta.map((row, t) => `<tr><td>t=${t + 1}</td>${row.map(val => `<td>${val.toExponential(4)}</td>`).join('')}</tr>`).join('')}
                </table>
            </div>

            <div class="table-wrapper">
                <h4>State Responsibilities (&#947;)</h4>
                <table>
                    <tr><th>Time Step</th>${states.map(s => `<th>${s}</th>`).join('')}</tr>
                    ${gamma.map((row, t) => `<tr><td>t=${t + 1}</td>${row.map(val => `<td>${val.toFixed(4)}</td>`).join('')}</tr>`).join('')}
                </table>
            </div>
        </div>
    `;

    // prepend the result
    iterationLog.innerHTML = html + iterationLog.innerHTML;
}

runIterBtn.addEventListener('click', () => runIteration());
runMultiBtn.addEventListener('click', () => {
    for (let i = 0; i < 5; i++) {
        runIteration();
    }
});
runConvBtn.addEventListener('click', () => {
    let delta = 1;
    let safeguard = 0;
    while (delta > 1e-4 && safeguard < 100) {
        delta = runIteration();
        safeguard++;
    }
});

function updateMetrics(ll, delta) {
    metricsPanel.style.display = 'flex';
    valIteration.innerText = iterationCount;
    valLogLikelihood.innerText = ll.toFixed(4);
    valDelta.innerText = delta === 0 && iterationCount === 1 ? "-" : (delta > 0 ? "+" : "") + delta.toFixed(6);
    valStatus.innerText = statusText;
    if (statusText === "Converged") {
        valStatus.style.color = "#10B981";
    } else {
        valStatus.style.color = "#F8FAFC";
    }
}

function updateChart(gamma) {
    const ctx = document.getElementById('gammaChart').getContext('2d');
    const T = sequenceInput.value.split(',').map(s => s.trim()).filter(s => s).length;

    // Prepare datasets for each state
    const datasets = states.map((state, i) => {
        // Generate a pseudo-random harmonious color based on index
        const hue = (i * 137.5) % 360;
        return {
            label: `State: ${state}`,
            data: gamma.map(row => row[i]),
            borderColor: `hsl(${hue}, 70%, 60%)`,
            backgroundColor: `hsla(${hue}, 70%, 60%, 0.2)`,
            borderWidth: 2,
            pointRadius: 4,
            fill: true,
            tension: 0.3
        };
    });

    const labels = Array.from({ length: T }, (_, i) => `t=${i + 1}`);

    if (gammaChartInstance) {
        gammaChartInstance.data.labels = labels;
        gammaChartInstance.data.datasets = datasets;
        gammaChartInstance.update();
    } else {
        gammaChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#94A3B8'
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: '#94A3B8'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: '#94A3B8'
                        }
                    }
                }
            }
        });
    }
}

function updateLLChart() {
    const ctx = document.getElementById('logLikelihoodChart').getContext('2d');
    const labels = Array.from({ length: llHistory.length }, (_, i) => `Iter ${i + 1}`);

    if (llChartInstance) {
        llChartInstance.data.labels = labels;
        llChartInstance.data.datasets[0].data = llHistory;
        llChartInstance.update();
    } else {
        llChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Log-Likelihood',
                    data: llHistory,
                    borderColor: '#8B5CF6',
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderWidth: 2,
                    pointRadius: 4,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#94A3B8'
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: '#94A3B8'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: '#94A3B8'
                        }
                    }
                }
            }
        });
    }
}

function updateNetwork(newA) {
    const container = document.getElementById('hmmNetwork');

    // Create nodes
    const nodes = states.map((state, i) => ({
        id: i,
        label: state,
        color: { background: `hsl(${(i * 137.5) % 360}, 70%, 60%)`, border: '#ffffff' },
        font: { color: '#ffffff' }
    }));

    // Create edges based on transition probabilities
    let edges = [];
    for (let i = 0; i < states.length; i++) {
        for (let j = 0; j < states.length; j++) {
            if (newA[i][j] > 0.01) { // Only draw if > 1%
                edges.push({
                    from: i,
                    to: j,
                    label: newA[i][j].toFixed(2),
                    arrows: 'to',
                    color: { color: 'rgba(255, 255, 255, 0.5)' },
                    font: { align: 'horizontal', color: '#94A3B8', strokeWidth: 0 },
                    width: newA[i][j] * 5,
                    smooth: { type: 'curvedCW', roundness: 0.2 }
                });
            }
        }
    }

    const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
    const options = {
        nodes: { shape: 'circle', margin: 10, shadow: true },
        edges: { font: { size: 12 }, shadow: true },
        physics: {
            hierarchicalRepulsion: {
                nodeDistance: 150
            }
        }
    };

    if (networkInstance) {
        networkInstance.setData(data);
    } else {
        networkInstance = new vis.Network(container, data, options);
    }
}

resetBtn.addEventListener('click', () => {
    iterationCount = 0;
    iterationLog.innerHTML = '';
    llHistory = [];
    lastLogLikelihood = null;
    statusText = "Initialized";
    resultsSection.style.display = 'none';
    metricsPanel.style.display = 'none';
    renderMatrices();
    if (gammaChartInstance) {
        gammaChartInstance.destroy();
        gammaChartInstance = null;
    }
    if (llChartInstance) {
        llChartInstance.destroy();
        llChartInstance = null;
    }
    if (networkInstance) {
        networkInstance.destroy();
        networkInstance = null;
    }
});
