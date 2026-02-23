"""
app.py — Production entry point (Zeabur / Docker / local).

The committed Dockerfile runs ``python app.py`` directly, which starts
the eventlet-based Socket.IO server without Gunicorn.  The ``PORT``
environment variable is respected (Zeabur injects it); defaults to 5000
for local development.
"""

import os
from hmm_service.app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    print(f"Checking if port {port} is in use...")
    import platform
    import subprocess
    try:
        if platform.system() == "Windows":
            cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
            subprocess.run(["powershell", "-Command", cmd], capture_output=True)
        else:
            cmd = f"lsof -ti:{port} | xargs kill -9"
            subprocess.run(cmd, shell=True, capture_output=True)
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("🚀 HMM Engine Server running!")
    print(f"👉 Dashboard available at: http://127.0.0.1:{port}/")
    print("=" * 50 + "\n")

    socketio.run(app, host="0.0.0.0", port=port, debug=False)
