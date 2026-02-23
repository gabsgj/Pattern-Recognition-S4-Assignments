from flask import Flask, render_template_string
import numpy as np
from hmm import HMM

app = Flask(__name__)
from state_transition_diagrams import create_blueprint
app.register_blueprint(create_blueprint(), url_prefix="/std")

# Train model once
observations = [0, 1, 0, 1, 1, 0]
n_states = 2
n_observations = 2

model = HMM(n_states, n_observations)
likelihoods = model.baum_welch(observations, max_iter=15)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>HMM Interactive Diagram</title>
    <link rel="stylesheet" href="/std/css/state_diagram.css">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="/std/js/state_diagram.js"></script>
</head>
<body>
<h2>HMM State Transition Diagram</h2>

<div id="diagram-container" class="std-canvas"></div>
<div id="inspector" class="std-inspector"></div>

<script>
const diagram = new StateTransitionDiagram('#diagram-container', '#inspector');

diagram.feedIteration({
    A: {{A}},
    B: {{B}},
    pi: {{pi}},
    iteration: 1,
    log_likelihood: {{likelihood}}
});

diagram.onComplete();
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(
        html_template,
        A=model.A.tolist(),
        B=model.B.tolist(),
        pi=model.pi.tolist(),
        likelihood=float(likelihoods[-1])
    )

if __name__ == "__main__":
    app.run(debug=True)