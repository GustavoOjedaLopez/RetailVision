# Gustavo Ojeda -- Predictive Assistance Alerts: web page that receives assistance alerts and shows them to staff
"""Small web server for staff alerts. Only the Python standard library is used (no installs).

Run it on the laptop:
    python -m scripts.alert_server
Then open http://localhost:8000 on the laptop, or http://<laptop-ip>:8000 on a phone
connected to the same Wi-Fi (the command prints the address; Windows may ask to allow
Python through the firewall: allow "private networks").

The live program sends alerts with send_alert() (HTTP POST /api/alerts). Staff press
"Attended" or "Not needed" on each alert; that answer is saved too, because it tells
us later which alerts were real (free labels to improve the model).

Only alert events are stored (time, camera, person ID, probability, staff answer) in
data/alerts.db. No images, no faces.

Try it without the camera:
    python -m scripts.alert_server --demo     (sends a fake alert every 20 s)
"""

import argparse
import json
import random
import socket
import sqlite3
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_DB = Path("data/alerts.db")
ANSWERS = ("attended", "not_needed")


class AlertStore:
    """Alerts saved in a small SQLite file; one lock because the web server uses several threads."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT NOT NULL, camera TEXT, track_id INTEGER, probability REAL,
            reason TEXT, answer TEXT, answered TEXT)""")
        self.db.commit()

    def add(self, camera, track_id, probability, reason):
        with self.lock:
            cursor = self.db.execute(
                "INSERT INTO alerts (created, camera, track_id, probability, reason) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), camera, track_id, probability, reason))
            self.db.commit()
            return cursor.lastrowid

    def answer(self, alert_id, answer):
        with self.lock:
            changed = self.db.execute("UPDATE alerts SET answer = ?, answered = ? WHERE id = ?",
                                      (answer, datetime.now().isoformat(timespec="seconds"), alert_id)).rowcount
            self.db.commit()
            return changed == 1

    def recent(self, limit=50):
        with self.lock:
            rows = self.db.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return None

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/alerts":
                self.send_json(store.recent())
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            data = self.read_json()
            if data is None:
                return self.send_json({"error": "invalid JSON"}, 400)
            if self.path == "/api/alerts":
                alert_id = store.add(str(data.get("camera", "cam0")), data.get("track_id"),
                                     data.get("probability"), str(data.get("reason", "")))
                print(f"Alert #{alert_id}: camera {data.get('camera')} ID {data.get('track_id')}")
                return self.send_json({"id": alert_id}, 201)
            parts = self.path.strip("/").split("/")  # api/alerts/<id>/answer
            if len(parts) == 4 and parts[:2] == ["api", "alerts"] and parts[3] == "answer" and parts[2].isdigit():
                if data.get("answer") not in ANSWERS:
                    return self.send_json({"error": f"answer must be one of {ANSWERS}"}, 400)
                ok = store.answer(int(parts[2]), data["answer"])
                return self.send_json({"ok": ok}, 200 if ok else 404)
            self.send_json({"error": "not found"}, 404)

        def log_message(self, *args):  # keep the terminal quiet; alerts are printed above
            pass

    return Handler


def send_alert(url, camera, track_id, probability, reason=""):
    """Send one alert to the server without stopping the camera loop (runs in a background thread).

    Gustavo Ojeda -- Predictive Assistance Alerts: if the server is off, the live program keeps working.
    """
    def post():
        payload = json.dumps({"camera": camera, "track_id": int(track_id),
                              "probability": round(float(probability), 3), "reason": reason}).encode()
        request = urllib.request.Request(url.rstrip("/") + "/api/alerts", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=2).close()
        except OSError as error:
            print(f"Could not send alert to {url}: {error}")

    threading.Thread(target=post, daemon=True).start()


def local_ip():
    """The laptop's address on the Wi-Fi, so a phone can open the page."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def demo_alerts(port):
    while True:
        time.sleep(20)
        send_alert(f"http://127.0.0.1:{port}", "aisle-1", random.randint(1, 60),
                   random.uniform(0.6, 0.95), "demo alert")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assistance Alerts</title>
<style>
  :root { --bg:#f5f6f8; --card:#fff; --text:#1d2330; --muted:#667085; --alert:#d92d20; --ok:#12b76a; --line:#e4e7ec; }
  @media (prefers-color-scheme: dark) { :root { --bg:#111418; --card:#1b2028; --text:#e8eaee; --muted:#98a2b3; --line:#2c333d; } }
  body { margin:0; font-family:system-ui, sans-serif; background:var(--bg); color:var(--text); }
  header { padding:16px; display:flex; align-items:center; justify-content:space-between; gap:8px; flex-wrap:wrap; }
  h1 { font-size:20px; margin:0; }
  #status { color:var(--muted); font-size:14px; }
  button { font-size:15px; padding:10px 14px; border-radius:10px; border:1px solid var(--line); background:var(--card); color:var(--text); cursor:pointer; }
  main { padding:0 16px 24px; max-width:720px; margin:0 auto; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; margin:10px 0; }
  .new { border:2px solid var(--alert); }
  .title { font-weight:700; font-size:17px; }
  .meta { color:var(--muted); font-size:14px; margin:4px 0 10px; }
  .actions { display:flex; gap:8px; flex-wrap:wrap; }
  .done { color:var(--muted); font-size:14px; }
  .attended { color:var(--ok); } .not_needed { color:var(--muted); }
  #empty { color:var(--muted); text-align:center; padding:40px 0; }
</style></head>
<body>
<header><h1>Assistance Alerts</h1>
  <span id="status">Connecting…</span>
  <button id="sound">Enable sound</button></header>
<main><div id="list"></div><div id="empty">No alerts yet.</div></main>
<script>
let lastId = null, soundOn = false, audio = null;
document.getElementById("sound").onclick = () => {
  audio = audio || new (window.AudioContext || window.webkitAudioContext)();
  soundOn = true; document.getElementById("sound").textContent = "Sound on"; beep();
};
function beep() {
  if (!soundOn || !audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = 880; o.connect(g); g.connect(audio.destination);
  g.gain.setValueAtTime(0.25, audio.currentTime); o.start(); o.stop(audio.currentTime + 0.35);
  if (navigator.vibrate) navigator.vibrate(300);
}
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
async function answer(id, value) {
  await fetch(`/api/alerts/${id}/answer`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({answer:value})});
  load();
}
async function load() {
  try {
    const alerts = await (await fetch("/api/alerts")).json();
    document.getElementById("status").textContent = "Live · updated " + new Date().toLocaleTimeString();
    document.getElementById("empty").style.display = alerts.length ? "none" : "block";
    if (alerts.length && lastId !== null && alerts[0].id > lastId) { beep(); document.title = "(!) Assistance Alerts"; }
    if (alerts.length) lastId = alerts[0].id;
    document.getElementById("list").innerHTML = alerts.map(a => `
      <div class="card ${a.answer ? "" : "new"}">
        <div class="title">Customer may need help · ${esc(a.camera)} · ID ${esc(a.track_id)}</div>
        <div class="meta">${esc(a.created.replace("T", " "))} · probability ${a.probability == null ? "–" : Math.round(a.probability * 100) + "%"}${a.reason ? " · " + esc(a.reason) : ""}</div>
        ${a.answer ? `<div class="done ${a.answer}">${a.answer === "attended" ? "Attended" : "Not needed"} · ${esc((a.answered || "").replace("T", " "))}</div>`
          : `<div class="actions"><button onclick="answer(${a.id}, 'attended')">Attended</button>
             <button onclick="answer(${a.id}, 'not_needed')">Not needed</button></div>`}
      </div>`).join("");
  } catch (e) { document.getElementById("status").textContent = "Server not reachable…"; }
}
window.onfocus = () => { document.title = "Assistance Alerts"; };
load(); setInterval(load, 2000);
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--demo", action="store_true", help="send a fake alert every 20 s")
    args = parser.parse_args()

    store = AlertStore(args.db)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(store))
    print(f"Alert page on this laptop: http://localhost:{args.port}")
    print(f"Alert page on a phone (same Wi-Fi): http://{local_ip()}:{args.port}")
    if args.demo:
        threading.Thread(target=demo_alerts, args=(args.port,), daemon=True).start()
        print("Demo mode: a fake alert every 20 s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
