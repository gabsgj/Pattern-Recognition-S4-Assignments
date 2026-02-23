from flask import Flask, render_template, jsonify, request
import numpy as np
from hmm import HMM
from state_transition_diagrams import create_blueprint   # 👈 add this

app = Flask(__name__)

# Serve JS/CSS assets at /std/...
app.register_blueprint(create_blueprint(), url_prefix="/std")  # 👈 add this

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    data = request.json
    seq = np.array(data["sequence"])
    states = int(data["states"])
    obs = int(data["observations"])

    model = HMM(states, obs)
    history = model.baum_welch(seq, max_iters=15)

    return jsonify(history)

if __name__ == "__main__":
    app.run(debug=True)