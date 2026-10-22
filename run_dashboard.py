"""
Financial Fraud Detection System — Dashboard Launcher
Launches the Streamlit application and automatically opens your web browser.
"""

import sys
import subprocess
import socket
import webbrowser
import time
from pathlib import Path

def find_available_port(start_port: int = 8501, max_attempts: int = 20) -> int:
    """Finds an open TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port

def main():
    root_dir = Path(__file__).parent.resolve()
    app_file = root_dir / "app.py"
    
    if not app_file.exists():
        print(f"[!] Error: app.py not found in {root_dir}")
        sys.exit(1)
        
    port = find_available_port(8501)
    url = f"http://localhost:{port}"
    
    print("=" * 72)
    print("  FINANCIAL FRAUD DETECTION SYSTEM | INTERACTIVE ANALYTICS DASHBOARD")
    print("  Zidio Development Internship Project Prototype")
    print("=" * 72)
    print(f"\n[+] Local Server URL: {url}")
    print(f"[+] Directory:        {root_dir}")
    print("\n[+] Launching Streamlit web application...")
    print("[+] Press Ctrl+C in this terminal window to stop the dashboard server.\n")
    print("=" * 72)
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    # Give server a moment then open browser
    def open_browser_later():
        time.sleep(2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser_later, daemon=True).start()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n[!] Dashboard server stopped.")
    except Exception as e:
        print(f"\n[!] Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
