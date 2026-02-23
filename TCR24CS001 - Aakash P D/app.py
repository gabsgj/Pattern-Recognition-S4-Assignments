from flask import Flask, render_template, request
import my_hmm as hmm
import diagram
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        obs_seq = request.form.get("obs", "").strip()
        hidden_input = request.form.get("hidden", "").strip()

        # Check if input is empty
        if not obs_seq or not hidden_input:
            return render_template("index.html", error="Please fill in all fields")

        # Convert hidden states to int
        try:
            hidden_states = int(hidden_input)
        except ValueError:
            return render_template("index.html", error="Hidden states must be an integer")

        # Run Baum-Welch
        A, B, pi, P_O = hmm.baum_welch(obs_seq, hidden_states)

        # Create state diagram
        diagram.create_diagram(A)

        return render_template("index.html", A=A, B=B, pi=pi, P_O=P_O)

    return render_template("index.html")
    

if __name__ == "__main__":
    import threading
    import webbrowser
    import time

    # Open the browser in a separate thread after a short delay
    def open_browser():
        time.sleep(1)  # wait 1 second for Flask to start
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser).start()
    app.run(debug=True)