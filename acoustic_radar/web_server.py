#!/usr/bin/env python3
"""
web_server.py — Local FastAPI + MJPEG view of the station's own display.

    Camera ──► Hailo ──► tracking ──► fusion ──► hud.render() ──┬──► cv2.imshow
                                                                └──► FrameBus
                                                                       │
                                                          FastAPI /video_feed
                                                                       │
                                                                  Ethernet
                                                                       │
                                                            http://<pi>:5000

═══════════════════════════════════════════════════════════════════
WHAT THIS MODULE DOES **NOT** DO
═══════════════════════════════════════════════════════════════════

It does not open a camera. It does not touch the Hailo device. It does not
run inference, tracking, DOA or fusion. It is a viewer: `publish()` hands it
the frame the station has ALREADY composed for its own window, and every
connected browser sees exactly what an operator standing at the Pi sees.

That is deliberate. A second camera handle would fail (picamera2 will not
open a sensor twice), a second Hailo VDevice would fail the same way, and a
per-client inference path would multiply the load by the number of tabs
open. There is one pipeline and one frame.

═══════════════════════════════════════════════════════════════════
LATEST-FRAME, NOT A QUEUE
═══════════════════════════════════════════════════════════════════

`FrameBus` holds exactly one frame. A browser slower than the pipeline
misses intermediate frames instead of falling further and further behind —
which is the correct behaviour for a live view, and the same rule
`target_state.LatestValue` already applies between the sensor threads.

There is no artificial frame-rate cap. A stream waits on a condition
variable and is woken the moment a new frame is published, so it runs as
fast as the pipeline, the JPEG encoder, the network and the browser allow.

═══════════════════════════════════════════════════════════════════
ONE ENCODE PER FRAME, NOT ONE PER CLIENT
═══════════════════════════════════════════════════════════════════

JPEG encoding is the most expensive thing this module does, so it is done
at most once per frame no matter how many browsers are watching, and NOT AT
ALL when nobody is connected. The encoded bytes are cached against the
frame's sequence number; the second and subsequent clients reuse them.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("station.web")


# ═══════════════════════════════════════════════════════════════
#  Rate measurement
# ═══════════════════════════════════════════════════════════════

class FpsMeter:
    """
    A measured rate, not a configured one.

    Every FPS this module reports is counted from real events over a real
    window. Nothing here is derived from a target, a sleep interval or a
    configuration value — the whole point of reporting four separate rates
    is to see WHICH stage is the bottleneck, and a number that came from a
    setting cannot show that.
    """

    __slots__ = ("_window_s", "_stamps", "_lock")

    def __init__(self, window_s: float = 2.0):
        self._window_s = float(window_s)
        self._stamps: List[float] = []
        self._lock = threading.Lock()

    def tick(self, n: int = 1) -> None:
        now = time.monotonic()
        with self._lock:
            self._stamps.extend([now] * int(n))
            cutoff = now - self._window_s
            if self._stamps[0] < cutoff:
                self._stamps = [t for t in self._stamps if t >= cutoff]

    @property
    def fps(self) -> float:
        now = time.monotonic()
        cutoff = now - self._window_s
        with self._lock:
            recent = [t for t in self._stamps if t >= cutoff]
            self._stamps = recent
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        if span <= 1e-6:
            return 0.0
        # n-1 intervals across the span — the rate of the samples we hold.
        return (len(recent) - 1) / span


# ═══════════════════════════════════════════════════════════════
#  The one-frame bus
# ═══════════════════════════════════════════════════════════════

class FrameBus:
    """
    The single latest annotated frame, plus the JPEG cache for it.

    Threading contract: `publish()` is called from the station's UI thread;
    `wait_for_frame()` and `encoded()` are called from uvicorn's worker
    threads. All shared state is guarded by one Condition — held only for
    reference assignments and a dictionary lookup, never across the encode.
    """

    def __init__(self, jpeg_quality: int = 80):
        self.jpeg_quality = int(jpeg_quality)
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        #: (seq, bytes) of the most recently encoded frame. Shared by every
        #: client so N browsers cost ONE encode, not N.
        self._encoded: Optional[Tuple[int, bytes]] = None
        #: Sequence number currently being encoded, so simultaneous clients
        #: wait for one encode instead of each doing their own.
        self._encoding_seq: Optional[int] = None
        self._closed = False

        self.jpeg_meter = FpsMeter()
        self.publish_meter = FpsMeter()
        #: One meter per live stream, so the reported MJPEG rate is a real
        #: per-client frame rate and not the sum across tabs.
        self._client_meters: Dict[int, FpsMeter] = {}
        self._next_client_id = 0

    # ── Producer side ──────────────────────────────────────────

    def publish(self, frame: np.ndarray) -> None:
        """
        Hand over the frame the station just composed.

        A reference is stored, not a copy: `hud.render()` reuses one canvas,
        so the buffer WILL be overwritten by the next render. The copy is
        taken in `encoded()` instead — under the lock, and only when a
        client actually needs it, so an idle server copies nothing.
        """
        with self._cond:
            if self._closed:
                return
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()
        self.publish_meter.tick()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    # ── Consumer side ──────────────────────────────────────────

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> int:
        """
        Block until a frame newer than `last_seq` exists. Returns its seq,
        or `last_seq` if the wait timed out (so the caller can re-check
        whether the client is still there).
        """
        with self._cond:
            if self._seq == last_seq and not self._closed:
                self._cond.wait(timeout)
            return self._seq

    def encoded(self, seq: int) -> Optional[bytes]:
        """
        JPEG bytes for `seq`, encoded EXACTLY ONCE however many clients ask.

        ⚠️ THE THUNDERING HERD. `publish()` calls notify_all(), so every
        connected stream wakes on the same frame at the same instant. An
        earlier version checked the cache, released the lock, and encoded —
        so all N clients missed the (still stale) cache together and all N
        encoded the same frame. Measured: 3 browsers turned 48 encodes/s
        into 113, tripling the most expensive work in this module for
        identical output.

        The first caller now claims the frame with `_encoding_seq` and the
        others WAIT on the condition for its result. The encode itself still
        happens outside the lock — holding it across cv2.imencode would
        stall the station's UI thread inside publish().
        """
        with self._cond:
            while True:
                if self._encoded is not None and self._encoded[0] == seq:
                    return self._encoded[1]
                if self._frame is None or self._seq != seq:
                    return None                      # frame already replaced
                if self._encoding_seq == seq:
                    # Another client is encoding this exact frame. Wait for
                    # it instead of duplicating the work.
                    if not self._cond.wait(0.5):
                        return None                  # encoder gave up
                    continue
                self._encoding_seq = seq
                frame = self._frame.copy()
                break

        data: Optional[bytes] = None
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                data = buf.tobytes()
            else:
                # Never silent: a failing encoder would otherwise look
                # exactly like a slow one.
                log.error("JPEG encoding failed for frame %d (%dx%d)",
                          seq, frame.shape[1], frame.shape[0])
        except cv2.error as exc:
            log.exception("JPEG encoding raised for frame %d: %s", seq, exc)
        finally:
            with self._cond:
                # Cleared even on failure, or every other client would wait
                # out the full timeout for a result that will never come.
                self._encoding_seq = None
                if data is not None:
                    self._encoded = (seq, data)
                self._cond.notify_all()

        if data is not None:
            self.jpeg_meter.tick()
        return data

    # ── Per-client bookkeeping ─────────────────────────────────

    def register_client(self) -> int:
        with self._cond:
            self._next_client_id += 1
            cid = self._next_client_id
            self._client_meters[cid] = FpsMeter()
        return cid

    def unregister_client(self, cid: int) -> None:
        with self._cond:
            self._client_meters.pop(cid, None)

    def client_tick(self, cid: int) -> None:
        meter = self._client_meters.get(cid)
        if meter is not None:
            meter.tick()

    @property
    def client_count(self) -> int:
        with self._cond:
            return len(self._client_meters)

    @property
    def mjpeg_fps(self) -> float:
        """The fastest live stream's real rate; 0.0 with nobody watching."""
        with self._cond:
            meters = list(self._client_meters.values())
        return max((m.fps for m in meters), default=0.0)

    @property
    def resolution(self) -> str:
        with self._cond:
            if self._frame is None:
                return "n/a"
            h, w = self._frame.shape[:2]
        return f"{w}x{h}"


# ═══════════════════════════════════════════════════════════════
#  Network helpers
# ═══════════════════════════════════════════════════════════════

def local_ipv4_addresses() -> List[str]:
    """
    Every IPv4 address this host answers on, best effort.

    Used only to PRINT a usable URL at start-up. The server itself always
    binds 0.0.0.0, so a wrong guess here cannot affect reachability — which
    is why this is allowed to be heuristic.
    """
    found: List[str] = []

    # The outbound-route trick: no packet is sent, the kernel just resolves
    # which interface would be used.
    for probe in ("192.168.50.2", "8.8.8.8"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.settimeout(0.2)
            sock.connect((probe, 9))
            ip = sock.getsockname()[0]
            if ip and ip not in found and not ip.startswith("127."):
                found.append(ip)
        except OSError:
            pass
        finally:
            sock.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            ip = info[4][0]
            if ip and ip not in found and not ip.startswith("127."):
                found.append(ip)
    except OSError as exc:
        log.debug("hostname lookup failed while listing addresses: %s", exc)

    return found


# ═══════════════════════════════════════════════════════════════
#  The page
# ═══════════════════════════════════════════════════════════════

_PAGE = """<!doctype html>
<title>Drone Detection Station</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;background:#12100e;color:#e6e3df;
      font:14px/1.45 ui-monospace,Menlo,Consolas,monospace}
 header{padding:10px 16px;border-bottom:1px solid #3d3529;
        display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:15px;margin:0;letter-spacing:.06em}
 .muted{color:#8b857c}
 main{display:flex;gap:16px;padding:16px;flex-wrap:wrap;align-items:flex-start}
 #view{background:#000;border:1px solid #3d3529;line-height:0;
       max-width:100%;overflow:hidden}
 #view img{display:block;max-width:100%;height:auto}
 aside{min-width:250px;flex:1}
 table{border-collapse:collapse;width:100%}
 th,td{padding:4px 8px;border-bottom:1px solid #2a251f;text-align:left}
 th{color:#8b857c;font-weight:400;white-space:nowrap}
 td{text-align:right;font-variant-numeric:tabular-nums}
 h2{font-size:12px;color:#8b857c;margin:18px 0 6px;letter-spacing:.1em;
    text-transform:uppercase;font-weight:400}
 .ok{color:#64dc82}.warn{color:#fabe3c}.bad{color:#ff5050}
</style>
<header>
  <h1>DRONE DETECTION STATION</h1>
  <span class="muted" id="url"></span>
</header>
<main>
  <div id="view"><img src="/video_feed" alt="live camera"></div>
  <aside>
    <h2>Pipeline rate</h2>
    <table>
      <tr><th>Camera</th><td id="camera_fps">-</td></tr>
      <tr><th>Hailo / YOLO</th><td id="hailo_fps">-</td></tr>
      <tr><th>JPEG encode</th><td id="jpeg_fps">-</td></tr>
      <tr><th>MJPEG out</th><td id="mjpeg_fps">-</td></tr>
    </table>
    <h2>Status</h2>
    <table>
      <tr><th>Hailo</th><td id="hailo">-</td></tr>
      <tr><th>Camera</th><td id="camera">-</td></tr>
      <tr><th>Microphone</th><td id="mic">-</td></tr>
      <tr><th>LED ring</th><td id="led">-</td></tr>
      <tr><th>System state</th><td id="state">-</td></tr>
      <tr><th>Detections</th><td id="det">-</td></tr>
      <tr><th>Bearing</th><td id="brg">-</td></tr>
      <tr><th>Resolution</th><td id="res">-</td></tr>
      <tr><th>Clients</th><td id="clients">-</td></tr>
    </table>
  </aside>
</main>
<script>
document.getElementById('url').textContent = location.origin;
function cls(v){return v==='ONLINE'?'ok':(v==='DEGRADED'||v==='STARTING')
  ?'warn':(v==='DISABLED'?'':'bad');}
function put(id,text,klass){var e=document.getElementById(id);
  e.textContent=text; e.className=klass||'';}
async function poll(){
  try{
    const r = await fetch('/status',{cache:'no-store'});
    const s = await r.json();
    put('camera_fps', s.camera_fps.toFixed(1)+' fps');
    put('hailo_fps',  s.hailo_fps.toFixed(1)+' fps');
    put('jpeg_fps',   s.jpeg_fps.toFixed(1)+' fps');
    put('mjpeg_fps',  s.mjpeg_fps.toFixed(1)+' fps');
    put('hailo', s.hailo, cls(s.hailo==='ONLINE'?'ONLINE':s.hailo));
    put('camera', s.camera_health, cls(s.camera_health));
    put('mic', s.acoustic_health, cls(s.acoustic_health));
    put('led', s.led, cls(s.led));
    put('state', s.state);
    put('det', String(s.detections));
    put('brg', s.bearing);
    put('res', s.resolution);
    put('clients', String(s.clients));
  }catch(e){ put('state','web server unreachable','bad'); }
}
poll(); setInterval(poll, 700);
</script>
"""


# ═══════════════════════════════════════════════════════════════
#  The server
# ═══════════════════════════════════════════════════════════════

class WebServer:
    """
    FastAPI + uvicorn on a daemon thread.

    Public surface used by main.py:
        start()               returns False if it cannot start
        publish(frame)        hand over the composed frame (UI thread)
        set_status_provider() callable returning the dashboard dict
        note_inference()      count one Hailo/YOLO invocation
        stop()

    Never raises into the station: a web-server failure must not take the
    detector down with it.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000,
                 jpeg_quality: int = 80):
        self.host = host
        self.port = int(port)
        self.bus = FrameBus(jpeg_quality=jpeg_quality)
        self.hailo_meter = FpsMeter()

        self._status_provider: Optional[Callable[[], dict]] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None            # uvicorn.Server
        self._started = threading.Event()
        self.error: Optional[str] = None

    # ── Wiring ─────────────────────────────────────────────────

    def set_status_provider(self, provider: Callable[[], dict]) -> None:
        self._status_provider = provider

    def publish(self, frame: np.ndarray) -> None:
        self.bus.publish(frame)

    def note_inference(self) -> None:
        """One Hailo/YOLO invocation actually happened."""
        self.hailo_meter.tick()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        try:
            app = self._build_app()
        except ImportError as exc:
            self.error = (f"{exc}. Install it inside the virtualenv: "
                          f"pip install fastapi uvicorn")
            log.error("[WEB] web interface unavailable: %s", self.error)
            return False
        except Exception as exc:
            self.error = str(exc)
            log.exception("[WEB] could not build the web application")
            return False

        import uvicorn

        config = uvicorn.Config(app, host=self.host, port=self.port,
                                log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._serve, name="web",
                                        daemon=True)
        self._thread.start()
        # Give the socket a moment to bind so a port clash is reported here
        # rather than silently a second later.
        self._started.wait(3.0)
        if self.error:
            return False

        log.info("[WEB] FastAPI server started")
        log.info("[WEB] Listening on %s:%d", self.host, self.port)
        addresses = local_ipv4_addresses()
        if addresses:
            for ip in addresses:
                marker = "  <-- expected Ethernet" if ip.startswith(
                    "192.168.50.") else ""
                log.info("[WEB] Local URL: http://%s:%d%s", ip, self.port,
                         marker)
                log.info("[WEB] MJPEG:     http://%s:%d/video_feed", ip,
                         self.port)
        else:
            log.warning("[WEB] could not determine any local IPv4 address — "
                        "the server is still listening on %s:%d",
                        self.host, self.port)
        return True

    def _serve(self) -> None:
        try:
            self._started.set()
            self._server.run()
        except OSError as exc:
            self.error = f"could not bind {self.host}:{self.port}: {exc}"
            log.error("[WEB] %s", self.error)
        except Exception as exc:
            self.error = str(exc)
            log.exception("[WEB] server thread crashed")
        finally:
            self._started.set()

    def stop(self, timeout: float = 3.0) -> None:
        # Close the bus FIRST so every MJPEG generator finishes on its own.
        # Without this they never return, uvicorn's graceful shutdown waits
        # for responses that will never end, and the forced exit tears the
        # streams down mid-write with a wall of CancelledError tracebacks.
        self.bus.close()
        time.sleep(0.15)
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            # Graceful first. uvicorn's graceful shutdown waits for open
            # responses to finish, and an MJPEG stream never finishes by
            # itself — so a still-watching browser would hold the station's
            # shutdown open indefinitely without the force below.
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.info("[WEB] a client is still streaming — forcing the "
                         "server to close")
                if self._server is not None:
                    self._server.force_exit = True
                self._thread.join(2.0)
            if self._thread.is_alive():
                log.warning("[WEB] server thread did not stop — leaving it "
                            "to the daemon shutdown")

    # ── Application ────────────────────────────────────────────

    def _build_app(self):
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.responses import StreamingResponse

        app = FastAPI(title="Drone Detection Station", docs_url=None,
                      redoc_url=None)
        bus = self.bus

        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse(_PAGE)

        @app.get("/health")
        def health() -> JSONResponse:
            return JSONResponse({"ok": True,
                                 "frames": bus.publish_meter.fps > 0.0})

        @app.get("/status")
        def status() -> JSONResponse:
            payload = {
                "camera_fps": 0.0, "hailo_fps": self.hailo_meter.fps,
                "jpeg_fps": bus.jpeg_meter.fps, "mjpeg_fps": bus.mjpeg_fps,
                "publish_fps": bus.publish_meter.fps,
                "resolution": bus.resolution, "clients": bus.client_count,
                "hailo": "UNKNOWN", "camera_health": "UNKNOWN",
                "acoustic_health": "UNKNOWN", "led": "UNKNOWN",
                "state": "-", "detections": 0, "bearing": "N/A",
            }
            if self._status_provider is not None:
                try:
                    payload.update(self._status_provider())
                except Exception as exc:
                    # Reported, never swallowed: a broken provider must not
                    # look like a healthy station with odd numbers.
                    log.exception("[WEB] status provider failed")
                    payload["state"] = f"status error: {exc}"
            return JSONResponse(payload)

        @app.get("/video_feed")
        def video_feed() -> StreamingResponse:
            return StreamingResponse(
                self._mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-store, no-cache, "
                                          "must-revalidate",
                         "Pragma": "no-cache",
                         "Connection": "close"})

        return app

    # ── MJPEG generator ────────────────────────────────────────

    def _mjpeg(self):
        """
        One multipart stream.

        Waits on the frame condition rather than sleeping a fixed interval,
        so the rate is whatever the pipeline can actually deliver — there is
        no artificial 20 or 30 fps ceiling anywhere in this path.
        """
        cid = self.bus.register_client()
        last_seq = -1
        log.info("[WEB] MJPEG client connected (%d now streaming)",
                 self.bus.client_count)
        try:
            while not self.bus.closed:
                seq = self.bus.wait_for_frame(last_seq, timeout=1.0)
                if seq == last_seq:
                    continue                     # timed out; re-check client
                data = self.bus.encoded(seq)
                if data is None:
                    continue
                last_seq = seq
                self.bus.client_tick(cid)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode() +
                       b"\r\n\r\n" + data + b"\r\n")
        except GeneratorExit:
            log.info("[WEB] MJPEG client disconnected")
            raise
        except Exception as exc:
            log.exception("[WEB] MJPEG stream failed: %s", exc)
        finally:
            self.bus.unregister_client(cid)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 66)
    print("web_server.py — self-test (no station, synthetic frames)")
    print("=" * 66)

    srv = WebServer(port=5000)
    srv.set_status_provider(lambda: {
        "camera_fps": 0.0, "hailo": "SELF-TEST", "camera_health": "SELF-TEST",
        "acoustic_health": "SELF-TEST", "led": "DISABLED",
        "state": "SELF-TEST", "detections": 0, "bearing": "N/A"})
    if not srv.start():
        raise SystemExit(f"could not start: {srv.error}")

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        i = 0
        while True:
            i += 1
            frame[:] = 30
            cv2.putText(frame, f"frame {i}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2)
            srv.publish(frame)
            srv.note_inference()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        srv.stop()
