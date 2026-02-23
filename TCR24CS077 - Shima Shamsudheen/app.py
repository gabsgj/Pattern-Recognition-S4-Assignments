from flask import Flask, render_template, request
import numpy as np
from src.hmm import HiddenMarkovModel
from src.baum_welch import baum_welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

def run_baum_welch_advanced(model, observations, max_iter=100, tol=1e-3):  # Changed to 1e-3
    """Run Baum-Welch with all tracking"""
    N = model.N
    M = model.M
    T = len(observations)
    
    # Storage for tracking
    likelihoods = []
    log_likelihoods = []
    alphas = []
    betas = []
    gammas = []
    A_history = []
    deltas = []
    
    print(f"\n🚀 Training HMM with {N} states, {M} symbols, {len(observations)} observations")
    
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
        
        # Store intermediate values for last iteration
        if iteration == max_iter - 1 or iteration == len(range(max_iter)) - 1:
            alphas = alpha[:15]
            betas = beta[:15]
            gammas = gamma[:15]
        
        # Compute likelihood
        likelihood = np.sum(alpha[-1])
        log_likelihood = np.log(likelihood + 1e-300)
        
        likelihoods.append(likelihood)
        log_likelihoods.append(log_likelihood)
        
        # Store A history
        A_history.append(model.A.copy())
        
        # Compute delta (change in log-likelihood)
        if iteration > 0:
            delta = abs(log_likelihood - log_likelihoods[-2])
        else:
            delta = 0
        deltas.append(delta)
        
        # Convergence check using log-likelihood
        if iteration > 0 and delta < tol:
            print(f"  ✓ CONVERGED at iteration {iteration + 1} (Δ = {delta:.6f} < {tol})")
            break
        
        # Compute xi
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            denom = np.sum(alpha[t][:, None] * model.A * model.B[:, observations[t+1]] * beta[t+1])
            if denom > 0:
                for i in range(N):
                    numer = alpha[t, i] * model.A[i, :] * model.B[:, observations[t+1]] * beta[t+1]
                    xi[t, i, :] = numer / denom
        
        # Update pi
        model.pi = gamma[0]
        
        # Update A
        for i in range(N):
            denom = np.sum(gamma[:-1, i])
            if denom > 0:
                for j in range(N):
                    model.A[i, j] = np.sum(xi[:, i, j]) / denom
        
        # Update B
        for i in range(N):
            denom = np.sum(gamma[:, i])
            if denom > 0:
                for k in range(M):
                    mask = (np.array(observations) == k)
                    model.B[i, k] = np.sum(gamma[mask, i]) / denom
        
        # Print progress every 5 iterations
        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}: Log-Likelihood = {log_likelihood:.4f}, Δ = {delta:.6f}")
    
    # Determine if converged (stopped early due to convergence)
    iterations_completed = len(likelihoods)
    converged = iterations_completed < max_iter
    
    print(f"\n✓ Training completed in {iterations_completed} iterations")
    print(f"✓ Status: {'CONVERGED' if converged else 'MAX ITERATIONS REACHED'}")
    print(f"✓ Final Log-Likelihood: {log_likelihoods[-1]:.4f}")
    print(f"✓ Final Δ: {deltas[-1]:.6f}")
    
    return {
        'model': model,
        'likelihoods': likelihoods,
        'log_likelihoods': log_likelihoods,
        'alphas': alphas,
        'betas': betas,
        'gammas': gammas,
        'A_history': A_history,
        'deltas': deltas,
        'iterations': iterations_completed,
        'converged': converged,
        'final_log_likelihood': log_likelihoods[-1],
        'final_likelihood': likelihoods[-1]
    }

def create_plots(result, hidden_states, observations):
    """Create all plots"""
    plots = {}
    
    # Original Likelihood Graph
    plt.figure(figsize=(10, 6))
    plt.plot(result['likelihoods'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("P(O | λ)", fontsize=12)
    plt.title("Baum-Welch Convergence", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/graph.png", dpi=100)
    plt.close()
    
    # Log-Likelihood Convergence
    plt.figure(figsize=(8, 4))
    plt.plot(result['log_likelihoods'], 'r-', linewidth=2, marker='s', markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Log P(O|λ)")
    plt.title("Log-Likelihood Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plots['log_likelihood'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Parameter Evolution
    if len(result['A_history']) > 1:
        plt.figure(figsize=(8, 4))
        A_history = np.array(result['A_history'])
        for i in range(hidden_states):
            for j in range(hidden_states):
                plt.plot(A_history[:, i, j], label=f'A[{i}][{j}]', linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Probability")
        plt.title("Transition Matrix Evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plots['param_evolution'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    
    # FSM Diagram
    create_fsm_diagram(result['model'])
    
    return plots

def create_fsm_diagram(model):
    """Create FSM diagram"""
    plt.figure(figsize=(6, 4))
    
    N = model.N
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radius = 0.3
    centers = [(0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)) for a in angles]
    
    # Draw states
    for i, (x, y) in enumerate(centers):
        circle = plt.Circle((x, y), 0.1, fill=True, color='lightblue', ec='black', linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(x, y, f'S{i}', ha='center', va='center', fontweight='bold')
    
    # Draw transitions
    for i in range(N):
        for j in range(N):
            if model.A[i, j] > 0.01:
                x1, y1 = centers[i]
                x2, y2 = centers[j]
                
                if i == j:
                    circle = plt.Circle((x1, y1 + 0.15), 0.08, fill=False, 
                                      ec='gray', linestyle='-', linewidth=1)
                    plt.gca().add_patch(circle)
                    plt.text(x1 + 0.12, y1 + 0.2, f'{model.A[i, j]:.2f}', 
                           fontsize=8, ha='center')
                else:
                    dx = x2 - x1
                    dy = y2 - y1
                    plt.arrow(x1 + 0.05*dx, y1 + 0.05*dy, 0.8*dx, 0.8*dy, 
                            head_width=0.03, head_length=0.03, fc='gray', ec='gray',
                            length_includes_head=True, alpha=0.5)
                    plt.text(x1 + 0.5*dx, y1 + 0.5*dy, f'{model.A[i, j]:.2f}', 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title("State Transition Diagram")
    
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/diagram.png", dpi=100, bbox_inches='tight')
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    plots = None
    error = None
    graph_path = None
    diagram_path = None

    if request.method == "POST":
        try:
            # Get form data
            hidden_states = int(request.form["hidden_states"])
            max_iter = int(request.form["max_iter"])
            
            # Parse observations
            observations = list(map(int, request.form["observations"].split(",")))
            
            # Optional symbols field
            if request.form.get("symbols") and request.form["symbols"].strip():
                M = int(request.form["symbols"])
            else:
                M = len(set(observations))  # Auto-detect
            
            # Create model
            model = HiddenMarkovModel(hidden_states, M)
            
            # Train with advanced tracking (tol=1e-3 for faster convergence)
            training_result = run_baum_welch_advanced(model, observations, max_iter, tol=1e-3)
            
            # Create intermediate table
            intermediate_table = []
            if len(training_result['alphas']) > 0:
                for t in range(min(10, len(training_result['alphas']))):
                    row = {
                        't': t,
                        'alpha': [f"{x:.4f}" for x in training_result['alphas'][t]],
                        'beta': [f"{x:.4f}" for x in training_result['betas'][t]],
                        'gamma': [f"{x:.4f}" for x in training_result['gammas'][t]]
                    }
                    intermediate_table.append(row)
            
            # Create plots
            plots = create_plots(training_result, hidden_states, observations)
            
            # Prepare results
            result = {
                "A": np.array2string(training_result['model'].A, precision=4, separator=', ', suppress_small=True),
                "B": np.array2string(training_result['model'].B, precision=4, separator=', ', suppress_small=True),
                "pi": np.array2string(training_result['model'].pi, precision=4, separator=', ', suppress_small=True),
                "likelihoods": training_result['likelihoods'],
                "final_likelihood": training_result['final_likelihood'],
                "iterations": training_result['iterations'],
                "final_log_likelihood": training_result['final_log_likelihood'],
                "converged": training_result['converged'],
                "delta": training_result['deltas'][-1] if training_result['deltas'] else 0,
                "intermediate_table": intermediate_table,
                "log_likelihoods": training_result['log_likelihoods']
            }
            
            graph_path = "static/graph.png"
            diagram_path = "static/diagram.png"
            
        except Exception as e:
            error = str(e)
            print(f"Error: {e}")
    
    return render_template("index.html", 
                          result=result, 
                          plots=plots, 
                          graph_path=graph_path,
                          diagram_path=diagram_path,
                          error=error,
                          request=request)

if __name__ == "__main__":
    app.run(debug=True)