"""
Root workspace launcher for EstateIQ House Price Prediction Web App.
Usage:
    python run.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "house_price_app")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app.app import app, _open_browser_when_ready

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    target_url = f"http://127.0.0.1:{port}"

    import threading
    threading.Thread(target=_open_browser_when_ready, args=[target_url], daemon=True).start()

    print("\n" + "=" * 65)
    print("  EstateIQ House Price Prediction Web App")
    print(f"  --> Status: LIVE & RUNNING")
    print(f"  --> Opening in browser automatically: {target_url}")
    print(f"  --> Or open manually: {target_url}")
    print("  --> (Keep this terminal window OPEN while browsing)")
    print("  --> Press CTRL+C to stop the server")
    print("=" * 65 + "\n")
    try:
        app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
    except OSError as e:
        if "10048" in str(e) or "already in use" in str(e).lower():
            fallback_port = 5001
            fallback_url = f"http://127.0.0.1:{fallback_port}"
            print(f"[!] Port {port} was already in use, switching to: {fallback_url}")
            threading.Thread(target=_open_browser_when_ready, args=[fallback_url], daemon=True).start()
            app.run(host="127.0.0.1", port=fallback_port, debug=True, use_reloader=False)
        else:
            raise e
