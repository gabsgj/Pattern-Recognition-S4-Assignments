from flask import Flask, render_template, request, jsonify
import numpy as np
import sys
import os
import traceback
import threading
import uuid
import time

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import HMM modules
try:
    from src.hmm import HiddenMarkovModel
    from src.baum_welch import baum_welch

    print("✓ Successfully imported HMM modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please ensure src folder has __init__.py, hmm.py, and baum_welch.py")
    sys.exit(1)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Store training progress for active sessions
training_sessions = {}


def create_fsm_diagram(model, filename='static/diagram.png'):
    """Create FSM diagram and save to file"""
    try:
        plt.figure(figsize=(12, 8))

        N = model.N
        if N == 2:
            # For 2 states, place them horizontally
            centers = [(0.3, 0.5), (0.7, 0.5)]
        else:
            # For more states, place in a circle
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radius = 0.3
            centers = [(0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)) for a in angles]

        # Draw states
        for i, (x, y) in enumerate(centers):
            circle = plt.Circle((x, y), 0.15, fill=True, color='#4a90e2', ec='#2c3e50', linewidth=2, alpha=0.8)
            plt.gca().add_patch(circle)
            plt.text(x, y, f'S{i}', ha='center', va='center', fontweight='bold', fontsize=14, color='white')

        # Draw transitions
        for i in range(N):
            for j in range(N):
                prob = model.A[i, j]
                if prob > 0.01:
                    x1, y1 = centers[i]
                    x2, y2 = centers[j]

                    if i == j:
                        # Self-loop
                        circle = plt.Circle((x1, y1 + 0.25), 0.1, fill=False,
                                            ec='#e74c3c', linestyle='-', linewidth=2, alpha=0.7)
                        plt.gca().add_patch(circle)
                        plt.text(x1 + 0.2, y1 + 0.3, f'{prob:.2f}',
                                 fontsize=11, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#e74c3c'))
                    else:
                        # Calculate arrow
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        dx, dy = dx / dist, dy / dist

                        start_x = x1 + dx * 0.15
                        start_y = y1 + dy * 0.15
                        end_x = x2 - dx * 0.15
                        end_y = y2 - dy * 0.15

                        plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                                  head_width=0.04, head_length=0.04, fc='#27ae60', ec='#27ae60',
                                  length_includes_head=True, alpha=0.7, linewidth=2)

                        mid_x = (start_x + end_x) / 2
                        mid_y = (start_y + end_y) / 2
                        plt.text(mid_x, mid_y, f'{prob:.2f}',
                                 fontsize=11, fontweight='bold', ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#27ae60'))

        # Add initial distribution
        for i, (x, y) in enumerate(centers):
            if model.pi[i] > 0.01:
                plt.annotate(f'π={model.pi[i]:.2f}', xy=(x - 0.1, y + 0.15), xytext=(x - 0.3, y + 0.25),
                             arrowprops=dict(arrowstyle='->', color='#f39c12', linewidth=2),
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#f39c12'))

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title("State Transition Diagram", fontsize=16, fontweight='bold', pad=20)

        os.makedirs('static', exist_ok=True)
        plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        return True

    except Exception as e:
        print(f"Error creating diagram: {e}")
        return False


def run_baum_welch_advanced(model, observations, max_iter=100, tol=1e-3, session_id=None):
    """Run Baum-Welch with tracking for all three plots"""
    try:
        N = model.N
        M = model.M
        T = len(observations)

        print(f"\n🚀 Training HMM with {N} states, {M} symbols, {T} observations")

        likelihoods = []
        log_likelihoods = []
        A_history = []  # Store transition matrix history for parameter evolution

        # Initialize session
        if session_id:
            training_sessions[session_id] = {
                'progress': [],
                'complete': False,
                'model': model,
                'A_history': [],
                'log_likelihoods': [],
                'likelihoods': []
            }

        for iteration in range(max_iter):
            # Forward & Backward
            alpha = model.forward(observations)
            beta = model.backward(observations)

            # Compute gamma
            gamma = np.zeros((T, N))
            for t in range(T):
                denom = np.sum(alpha[t] * beta[t])
                if denom > 0:
                    gamma[t] = (alpha[t] * beta[t]) / denom

            # Compute likelihood
            likelihood = np.sum(alpha[-1])
            log_likelihood = np.log(likelihood + 1e-300)

            likelihoods.append(float(likelihood))
            log_likelihoods.append(float(log_likelihood))

            # Store A history for parameter evolution plot
            A_history.append(model.A.copy())

            # Store history in session
            if session_id:
                training_sessions[session_id]['A_history'] = [a.tolist() for a in A_history]
                training_sessions[session_id]['log_likelihoods'] = log_likelihoods.copy()
                training_sessions[session_id]['likelihoods'] = likelihoods.copy()

            # Check convergence
            delta = 0
            if iteration > 0:
                delta = abs(log_likelihood - log_likelihoods[-2])
                if delta < tol:
                    print(f"✓ Converged at iteration {iteration + 1}")
                    break

            # Compute xi
            xi = np.zeros((T - 1, N, N))
            for t in range(T - 1):
                denom = np.sum(alpha[t][:, None] * model.A * model.B[:, observations[t + 1]] * beta[t + 1])
                if denom > 0:
                    for i in range(N):
                        numer = alpha[t, i] * model.A[i, :] * model.B[:, observations[t + 1]] * beta[t + 1]
                        xi[t, i, :] = numer / denom

            # Update parameters
            model.pi = gamma[0].copy()

            for i in range(N):
                denom = np.sum(gamma[:-1, i])
                if denom > 0:
                    for j in range(N):
                        model.A[i, j] = np.sum(xi[:, i, j]) / denom

            for i in range(N):
                denom = np.sum(gamma[:, i])
                if denom > 0:
                    for k in range(M):
                        mask = (np.array(observations) == k)
                        if np.any(mask):
                            model.B[i, k] = np.sum(gamma[mask, i]) / denom

            # Save snapshot every iteration
            if session_id:
                snapshot = {
                    'iteration': iteration + 1,
                    'log_likelihood': float(log_likelihood),
                    'delta': float(delta),
                    'A': model.A.tolist(),
                    'B': model.B.tolist(),
                    'pi': model.pi.tolist(),
                    'log_likelihoods': [float(x) for x in log_likelihoods],
                    'likelihoods': [float(x) for x in likelihoods],
                    'A_history': [a.tolist() for a in A_history],
                    'converged': False,
                    'complete': False
                }
                training_sessions[session_id]['progress'].append(snapshot)

            if (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration + 1}: Log-Likelihood = {log_likelihood:.4f}")

        # Training complete
        iterations_completed = len(likelihoods)
        converged = iterations_completed < max_iter

        # Generate diagram
        if session_id:
            try:
                diagram_file = f'static/diagram_{session_id}.png'
                if create_fsm_diagram(model, diagram_file):
                    training_sessions[session_id]['diagram'] = f'/static/diagram_{session_id}.png'
            except Exception as e:
                print(f"Error generating diagram: {e}")

            # Final snapshot with all data
            final_snapshot = {
                'iteration': iterations_completed,
                'log_likelihood': float(log_likelihoods[-1]),
                'delta': float(delta),
                'A': model.A.tolist(),
                'B': model.B.tolist(),
                'pi': model.pi.tolist(),
                'log_likelihoods': [float(x) for x in log_likelihoods],
                'likelihoods': [float(x) for x in likelihoods],
                'A_history': [a.tolist() for a in A_history],
                'converged': converged,
                'complete': True
            }
            training_sessions[session_id]['progress'].append(final_snapshot)
            training_sessions[session_id]['complete'] = True

        print(f"\n✓ Training completed in {iterations_completed} iterations")
        print(f"✓ Final Log-Likelihood: {log_likelihoods[-1]:.4f}")
        return model, likelihoods

    except Exception as e:
        print(f"Error in training: {e}")
        traceback.print_exc()
        return None, None


@app.route("/api/progress/<session_id>")
def get_progress(session_id):
    """Get training progress"""
    session = training_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify({
        'progress': session['progress'],
        'complete': session['complete']
    })


@app.route("/api/diagram/<session_id>")
def get_diagram(session_id):
    """Get diagram path"""
    session = training_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    diagram_path = session.get('diagram')
    if not diagram_path:
        return jsonify({'error': 'Diagram not ready'}), 404

    return jsonify({'diagram': diagram_path})


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    session_id = None

    if request.method == "POST":
        try:
            # Get form data
            hidden_states = int(request.form["hidden_states"])
            max_iter = int(request.form["max_iter"])

            # Parse observations
            obs_str = request.form["observations"].strip()
            observations = [int(x.strip()) for x in obs_str.split(",")]

            # Determine M
            if request.form.get("symbols") and request.form["symbols"].strip():
                M = int(request.form["symbols"])
            else:
                M = len(set(observations))

            print(f"\n📝 Training started:")
            print(f"  States: {hidden_states}, Symbols: {M}, Iterations: {max_iter}")
            print(f"  Observations: {observations}")

            # Create model
            model = HiddenMarkovModel(hidden_states, M)

            # Generate session ID
            session_id = str(uuid.uuid4())

            # Start training thread
            thread = threading.Thread(
                target=run_baum_welch_advanced,
                args=(model, observations, max_iter, 1e-3, session_id)
            )
            thread.daemon = True
            thread.start()

            return render_template("index.html",
                                   session_id=session_id,
                                   error=None)

        except Exception as e:
            error = str(e)
            print(f"Error: {e}")
            traceback.print_exc()

    return render_template("index.html", session_id=None, error=error)


if __name__ == "__main__":
    print("\n🚀 Starting HMM Baum-Welch Server...")
    print("📡 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, threaded=True)