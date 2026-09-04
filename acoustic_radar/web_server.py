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

from latency import BUDGET

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
        #: Age of the camera frame this composed image was built from, at the
        #: moment it was published. Carried per-frame into the MJPEG part
        #: headers so the BROWSER can add its own leg and report a real
        #: capture -> display latency instead of the server guessing one.
        self._frame_age_ms: Optional[float] = None
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

    def publish(self, frame: np.ndarray,
                frame_age_ms: Optional[float] = None) -> None:
        """
        Hand over the frame the station just composed.

        A reference is stored, not a copy, and that is safe because `hud`
        alternates between TWO canvases: the buffer handed over here is not
        the one the next render() writes into, so a consumer has a full frame
        period to take its copy. (It was not safe when the HUD reused a single
        canvas — see the note in hud.HUD.__init__.) The copy is still taken in
        `encoded()`, only when a client actually needs it, so an idle server
        copies nothing.
        """
        with self._cond:
            if self._closed:
                return
            self._frame = frame
            self._frame_age_ms = frame_age_ms
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

    def frame_age_ms(self) -> Optional[float]:
        with self._cond:
            return self._frame_age_ms

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
        _t_enc = time.monotonic()
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                data = buf.tobytes()
                # Measured on the thread that actually encodes, so the cost of
                # the single most expensive operation in this module is a
                # number rather than an opinion.
                BUDGET.record("jpeg_encode",
                              (time.monotonic() - _t_enc) * 1000.0)
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

# Raw string: the JavaScript below contains regular expressions with \s and
# \d, which Python would otherwise try to interpret as its own escapes.
_PAGE = r"""<!doctype html>
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
 .src{font-size:10px;color:#6d675e;letter-spacing:.04em}
 #view canvas,#view img{display:block;max-width:100%;height:auto}
</style>
<header>
  <h1>DRONE DETECTION STATION</h1>
  <span class="muted" id="url"></span>
</header>
<main>
  <div id="view"><canvas id="cv" width="640" height="562"></canvas></div>
  <aside>
    <h2>Pipeline rate</h2>
    <table>
      <tr><th>Camera <span class="src">sensor</span></th>
          <td id="camera_fps">-</td></tr>
      <tr><th>Processing loop <span class="src">worker</span></th>
          <td id="process_fps">-</td></tr>
      <tr><th>Hailo / YOLO <span class="src">around infer</span></th>
          <td id="hailo_fps">-</td></tr>
      <tr><th>JPEG encode <span class="src">encoder</span></th>
          <td id="jpeg_fps">-</td></tr>
      <tr><th>MJPEG out <span class="src">socket</span></th>
          <td id="mjpeg_fps">-</td></tr>
      <tr><th>Received <span class="src">browser</span></th>
          <td id="recv_fps">-</td></tr>
      <tr><th>Displayed <span class="src">browser</span></th>
          <td id="disp_fps">-</td></tr>
    </table>
    <h2>Latency</h2>
    <table>
      <tr><th>Inference</th><td id="infer_ms">-</td></tr>
      <tr><th>Capture wait</th><td id="cap_ms">-</td></tr>
      <tr><th>Frame age at publish</th><td id="age_ms">-</td></tr>
      <tr><th>Capture &rarr; display</th><td id="lat_ms">-</td></tr>
      <tr><th>Frames dropped</th><td id="dropped">-</td></tr>
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
  if(!e) return; e.textContent=text; e.className=klass||'';}
function fps(v){return (v===null||v===undefined)?'-':v.toFixed(1)+' fps';}
function ms(v){return (v===null||v===undefined)?'-':v.toFixed(0)+' ms';}

/* ══════════════════════════════════════════════════════════════════
   REAL DISPLAY FPS — measured HERE, because only here can it be.
   ══════════════════════════════════════════════════════════════════
   The server knows when it wrote bytes to a socket. It cannot know when
   this browser decoded them or when the compositor painted them, and for a
   slow client those differ without bound. So:

     RECEIVED  = multipart parts fully read off the ONE existing MJPEG
                 connection. No second stream, no second encode: this reads
                 exactly the response an <img> would have consumed.
     DISPLAYED = frames actually drawn, counted inside requestAnimationFrame,
                 which fires only when the browser really composites.
     DROPPED   = gaps in the server's X-Frame-Seq, so frames the pipeline
                 produced but this client never got are visible instead of
                 silently improving the received rate.

   If streaming fetch is unavailable (or fails), we fall straight back to a
   plain <img src="/video_feed"> and report the two browser numbers as
   UNAVAILABLE rather than substituting a server-side number under a
   browser-side name.
================================================================== */
var recvTimes = [], paintTimes = [], latSamples = [];
var lastSeq = null, dropped = 0, pending = null, streaming = false;
var canvas = document.getElementById('cv'), ctx = canvas.getContext('2d');

function rate(times){
  var now = performance.now(), cut = now - 2000, i = 0;
  while(i < times.length && times[i] < cut) i++;
  times.splice(0, i);
  if(times.length < 2) return 0;
  var span = times[times.length-1] - times[0];
  return span > 0 ? (times.length - 1) * 1000 / span : 0;
}

function paint(){
  if(pending){
    var bmp = pending; pending = null;
    if(canvas.width !== bmp.width || canvas.height !== bmp.height){
      canvas.width = bmp.width; canvas.height = bmp.height;
    }
    ctx.drawImage(bmp, 0, 0);
    if(bmp.close) bmp.close();
    paintTimes.push(performance.now());
  }
  requestAnimationFrame(paint);
}

function fallbackToImg(why){
  streaming = false;
  var view = document.getElementById('view');
  view.innerHTML = '<img src="/video_feed" alt="live camera">';
  put('recv_fps','UNAVAILABLE'); put('disp_fps','UNAVAILABLE');
  put('lat_ms','UNAVAILABLE'); put('dropped','UNAVAILABLE');
  console.warn('display metrics unavailable, using <img>:', why);
}

/* Part headers are always well under 256 bytes, so the search for their
   terminator is bounded. Scanning the whole buffer instead would re-scan the
   entire JPEG body on every arriving chunk — O(n^2) in the frame size, paid
   on the viewer's CPU for no benefit. */
var HDR_SCAN_MAX = 512;
function indexOfSeq(buf, pat, from){
  var end = Math.min(buf.length, from + HDR_SCAN_MAX) - pat.length;
  outer: for(var i = from; i <= end; i++){
    for(var j = 0; j < pat.length; j++) if(buf[i+j] !== pat[j]) continue outer;
    return i;
  }
  return -1;
}

async function stream(){
  if(!window.ReadableStream || !window.fetch || !window.createImageBitmap){
    fallbackToImg('browser lacks streaming fetch or createImageBitmap');
    return;
  }
  streaming = true;
  requestAnimationFrame(paint);
  try{
    const resp = await fetch('/video_feed', {cache:'no-store'});
    if(!resp.body) throw new Error('no response body');
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    const CRLF2 = [13,10,13,10];
    let buf = new Uint8Array(0);
    for(;;){
      const {done, value} = await reader.read();
      if(done) break;
      const merged = new Uint8Array(buf.length + value.length);
      merged.set(buf); merged.set(value, buf.length);
      buf = merged;
      for(;;){
        const hdrEnd = indexOfSeq(buf, CRLF2, 0);
        if(hdrEnd < 0){
          /* No header terminator within the bounded scan window means this
             is not the stream we expect. Fall back rather than buffer
             without limit on the viewer's machine. */
          if(buf.length > HDR_SCAN_MAX){
            fallbackToImg('malformed multipart headers');
            return;
          }
          break;
        }
        const head = dec.decode(buf.subarray(0, hdrEnd));
        const mLen = /content-length:\s*(\d+)/i.exec(head);
        if(!mLen) break;
        const len = parseInt(mLen[1], 10);
        const start = hdrEnd + 4;
        if(buf.length < start + len) break;
        const jpeg = buf.slice(start, start + len);
        buf = buf.slice(start + len);

        const mSeq = /x-frame-seq:\s*(\d+)/i.exec(head);
        const mAge = /x-frame-age-ms:\s*(-?[\d.]+)/i.exec(head);
        const mSrv = /x-server-ms:\s*([\d.]+)/i.exec(head);
        if(mSeq){
          const seq = parseInt(mSeq[1], 10);
          if(lastSeq !== null && seq > lastSeq + 1) dropped += seq - lastSeq - 1;
          lastSeq = seq;
        }
        recvTimes.push(performance.now());
        /* capture -> display = how old the frame already was when the server
           published it, PLUS how long it took to reach and decode here.
           The second leg uses the server's own wall clock in the header
           against ours; a client whose clock is offset from the Pi's would
           bias it, so it is only reported when the result is sane. */
        const arrivedWall = Date.now();
        try{
          const bmp = await createImageBitmap(new Blob([jpeg], {type:'image/jpeg'}));
          pending = bmp;
          if(mAge && mSrv){
            const age = parseFloat(mAge[1]);
            const net = arrivedWall - parseFloat(mSrv[1]);
            if(age >= 0 && net >= 0 && net < 5000) latSamples.push(age + net);
            if(latSamples.length > 60) latSamples.shift();
          }
        }catch(e){ /* one corrupt part must not kill the stream */ }
      }
    }
    fallbackToImg('stream ended');
  }catch(e){
    fallbackToImg(e);
  }
}

async function poll(){
  try{
    const r = await fetch('/status',{cache:'no-store'});
    const s = await r.json();
    /* Labelled by WHERE it was measured. camera_fps is the sensor's own rate
       via SensorTimestamp; if the driver does not report it the server says
       so and we say so too, rather than showing the loop rate as if it were
       the camera. */
    put('camera_fps', fps(s.camera_fps) +
        (s.camera_fps_is_sensor ? '' : '  (loop, no SensorTimestamp)'),
        s.camera_fps_is_sensor ? '' : 'warn');
    put('process_fps', fps(s.process_fps));
    put('hailo_fps',  fps(s.hailo_fps));
    put('jpeg_fps',   fps(s.jpeg_fps));
    put('mjpeg_fps',  fps(s.mjpeg_fps));
    put('infer_ms',   ms(s.inference_ms));
    put('cap_ms',     ms(s.capture_wait_ms));
    put('age_ms',     ms(s.frame_age_ms),
        (s.frame_age_ms !== null && s.frame_age_ms > 100) ? 'warn' : '');
    if(streaming){
      const rf = rate(recvTimes), df = rate(paintTimes);
      put('recv_fps', fps(rf));
      put('disp_fps', fps(df));
      put('dropped', String(dropped));
      let lat = null;
      if(latSamples.length){
        lat = latSamples.slice().sort((a,b)=>a-b)[Math.floor(latSamples.length/2)];
      }
      put('lat_ms', lat === null ? 'measuring...' : ms(lat));
      fetch('/client_metrics', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({received_fps: rf, displayed_fps: df,
                              display_latency_ms: lat === null ? 0 : lat,
                              dropped: dropped})}).catch(()=>{});
    }
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
stream();
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

        # ── Client-reported display metrics ──
        #
        # ⚠️ THE SERVER CANNOT MEASURE THESE, AND MUST NOT PRETEND TO.
        #
        # Everything this process can observe stops at the socket: it knows
        # when it handed bytes to the kernel, not when a browser decoded them
        # and not when a compositor put them on a screen. Those can differ by
        # an unbounded amount for a slow client or a slow link.
        #
        # So the browser measures its own two numbers and posts them back
        # here. They are stored verbatim, tagged as client-reported, and go
        # stale on their own if the browser stops reporting — a number nobody
        # is currently producing must not keep being displayed as current.
        self._client_metrics: Dict[str, float] = {}
        self._client_metrics_stamp = 0.0

        self._thread: Optional[threading.Thread] = None
        self._server = None            # uvicorn.Server
        self._started = threading.Event()
        self.error: Optional[str] = None

    # ── Wiring ─────────────────────────────────────────────────

    def set_status_provider(self, provider: Callable[[], dict]) -> None:
        self._status_provider = provider

    def publish(self, frame: np.ndarray,
                frame_age_ms: Optional[float] = None) -> None:
        self.bus.publish(frame, frame_age_ms)

    def note_inference(self) -> None:
        """
        One Hailo/YOLO invocation actually happened.

        ⚠️ NO LONGER CALLED BY main.py, and deliberately so — see the note in
        Station.run(). The station's UI loop could only observe a subset of
        camera frames, so counting there produced a rate capped by
        ui.max_ui_fps rather than the detector's real rate. The real rate is
        now measured inside camera_worker around the forward pass itself and
        arrives through the status provider. This method is kept for the
        module self-test and for any caller that genuinely is on the
        inference path.
        """
        self.hailo_meter.tick()

    def note_client_metrics(self, metrics: Dict[str, float]) -> None:
        """Store what a browser reported about its own display performance."""
        self._client_metrics = dict(metrics)
        self._client_metrics_stamp = time.monotonic()

    def client_metrics(self, max_age_s: float = 5.0) -> Dict[str, float]:
        """Client metrics, or {} if nothing recent was reported."""
        if not self._client_metrics:
            return {}
        if time.monotonic() - self._client_metrics_stamp > max_age_s:
            return {}
        return dict(self._client_metrics)

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
        from fastapi import FastAPI, Request
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
                # Defaults only. The station's provider overwrites camera_fps
                # and hailo_fps with values measured at the camera worker; the
                # hailo_meter fallback exists for the standalone self-test.
                "camera_fps": 0.0, "camera_fps_is_sensor": False,
                "process_fps": 0.0, "hailo_fps": self.hailo_meter.fps,
                "frame_age_ms": None, "capture_wait_ms": None,
                "inference_ms": None,
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
            # Merged last and namespaced, so a client-reported number can
            # never be mistaken for one this process measured.
            payload["client"] = self.client_metrics()
            return JSONResponse(payload)

        @app.post("/client_metrics")
        async def client_metrics(request: Request) -> JSONResponse:
            """
            A browser reporting its OWN display performance.

            `async def` on purpose: the only awaited call is reading a request
            body of a few dozen bytes, and there is no blocking work here at
            all, so a threadpool slot would be pure overhead. Every other
            endpoint in this file is `def` precisely because it CAN block.

            Untrusted input: only known keys are kept, only finite numbers are
            accepted, and nothing here can influence the pipeline — these
            values are displayed and nothing else reads them.
            """
            try:
                raw = await request.json()
            except Exception:
                return JSONResponse({"ok": False, "error": "bad json"},
                                    status_code=400)
            if not isinstance(raw, dict):
                return JSONResponse({"ok": False, "error": "expected object"},
                                    status_code=400)
            allowed = ("received_fps", "displayed_fps", "decoded_fps",
                       "display_latency_ms", "dropped")
            clean: Dict[str, float] = {}
            for key in allowed:
                value = raw.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = float(value)
                    if value == value and abs(value) != float("inf"):
                        clean[key] = value
            self.note_client_metrics(clean)
            return JSONResponse({"ok": True})

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
                # ── Per-part metadata ──
                #
                # X-Frame-Seq lets the browser count DROPPED frames (gaps in
                # the sequence) rather than only the ones it received, and
                # X-Frame-Age-Ms carries how old the camera frame already was
                # when it was published. The browser adds its own leg — encode
                # queue, network, decode, paint — and reports the total back,
                # which is the only place a real capture-to-display latency
                # can be assembled.
                #
                # These are ordinary multipart part headers: a browser
                # rendering the stream in an <img> ignores them completely, so
                # the plain fallback path is unaffected.
                age = self.bus.frame_age_ms()
                age_header = (b"X-Frame-Age-Ms: "
                              + (b"%.1f" % age if age is not None else b"-1")
                              + b"\r\n")
                # ⚠️ time.time(), not time.monotonic() — the ONE place in this
                # project where wall clock is correct. This value is compared
                # against the BROWSER's Date.now(), on a different machine; a
                # monotonic clock is meaningless across processes, let alone
                # across hosts. The JS side discards the result when it comes
                # out negative or absurd, which is what an unsynchronised
                # client clock produces, so a skewed clock degrades the
                # latency figure to "unavailable" instead of to a wrong number.
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"X-Frame-Seq: " + str(seq).encode() + b"\r\n" +
                       age_header +
                       b"X-Server-Ms: " +
                       ("%.1f" % (time.time() * 1000.0)).encode() + b"\r\n"
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
