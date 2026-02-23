let diagram = null;

window.onload = () => {
    diagram = new StateTransitionDiagram('#diagram-container', '#inspector', {
        particlesEnabled: true,
        decongestionEnabled: false,
        animationSpeed: 1
    });

    diagram.wireControls({
        btnFirst: document.getElementById('btn-first'),
        btnBack: document.getElementById('btn-back'),
        btnPlay: document.getElementById('btn-play'),
        btnForward: document.getElementById('btn-fwd'),
        btnLast: document.getElementById('btn-last'),
        speedSelect: document.getElementById('speed-select'),
        timeline: document.getElementById('timeline'),
        iterLabel: document.getElementById('iter-label'),
        btnParticles: document.getElementById('btn-particles'),
        btnDecongestion: document.getElementById('btn-decongestion'),
    });
};

function train() {
    const seq = document.getElementById("sequence").value.split(",").map(Number);
    const states = document.getElementById("states").value;
    const obs = document.getElementById("obs").value;

    fetch("/train", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({sequence: seq, states: states, observations: obs})
    })
    .then(res => res.json())
    .then(history => {
        diagram.reset();

        history.forEach(iter => {
            diagram.feedIteration({
                A: iter.A,
                B: iter.B,
                pi: iter.pi,
                iteration: iter.iteration,
                log_likelihood: iter.log_likelihood
            });
        });

        diagram.onComplete();

        // 🔥 Draw convergence graph
        drawConvergenceChart(history);
    });
}
function drawConvergenceChart(data) {
    const svg = d3.select("#convergence-chart");
    svg.selectAll("*").remove();

    const width = 900;
    const height = 350;
    const margin = { top: 20, right: 30, bottom: 40, left: 70 };

    const x = d3.scaleLinear()
        .domain([1, data.length])
        .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
        .domain([
            d3.min(data, d => d.log_likelihood) - 0.5,
            d3.max(data, d => d.log_likelihood) + 0.5
        ])
        .range([height - margin.bottom, margin.top]);

    const line = d3.line()
        .x((d, i) => x(i + 1))
        .y(d => y(d.log_likelihood))
        .curve(d3.curveMonotoneX);

    // Axes
    svg.append("g")
        .attr("transform", `translate(0, ${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(data.length));

    svg.append("g")
        .attr("transform", `translate(${margin.left}, 0)`)
        .call(d3.axisLeft(y));

    // Line
    svg.append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "#2563eb")
        .attr("stroke-width", 2.5)
        .attr("d", line);

    // Points
    svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", (d, i) => x(i + 1))
        .attr("cy", d => y(d.log_likelihood))
        .attr("r", 4)
        .attr("fill", "#ef4444");

    // Labels
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", height - 5)
        .attr("text-anchor", "middle")
        .text("Iteration");

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .text("Log-Likelihood");
}