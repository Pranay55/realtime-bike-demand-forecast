import sys
from pathlib import Path
import threading

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(project_root / "src"))
sys.path.insert(0, str(project_root))

from app_ui.app import app
from entrypoints.inference import run_inference

def start_inference():
    run_inference()

if __name__ == "__main__":
    threading.Thread(target=start_inference, daemon=True).start()

    app.run(debug=False, host="0.0.0.0", port=8050)