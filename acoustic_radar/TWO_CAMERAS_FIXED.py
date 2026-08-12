import time
import inspect
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional
from camera_manager import CameraManager

# Boundary requested: closer than 1.5 m -> NEAR, farther -> FAR.
#
# The two numbers are deliberately NOT both 1.5. A single shared boundary
# makes the system oscillate: at exactly 1.5 m the measured distance
# jitters by a few centimetres frame to frame, so it would switch camera,
# immediately qualify to switch back, and keep doing that (limited only
# by the 0.5 s debounce). The 0.5 m gap means: switch to NEAR once the
# drone is genuinely inside 1.5 m, and only return to FAR once it has
# clearly left, past 2.0 m. Between the two it simply keeps whichever
# camera it already has.
SWITCH_TO_NEAR_BELOW_M = 1.5
SWITCH_TO_FAR_ABOVE_M  = 2.0

# FALLBACK, used only when the active camera has no focal calibration
# (CAMERA_FOCAL_PX entry is None). These are the original pixel-height
# thresholds, kept so an uncalibrated system still switches at all.
THRESHOLD_NEAR_HEIGHT = 120
THRESHOLD_FAR_HEIGHT = 80
# INTEGRATION CHANGE (unified system): the Hailo runtime is imported
# lazily instead of at module import time.
#
# Rationale: the unified application imports this module for its tracker and
# inference classes. A hard top-level import means that on any machine
# without the Hailo runtime — a dev laptop, or a Pi whose PCIe driver failed
# to load — `import TWO_CAMERAS_FIXED` raises ImportError and takes down the
# ACOUSTIC subsystem too. The integration is required to isolate that
# failure: no Hailo must degrade to "camera offline", not "system dead".
#
# On a working Pi nothing changes: the same names are bound, just at
# HailoInference() construction time. The names are also still module-level
# globals afterwards, so any existing code referencing them keeps working.
HEF = VDevice = HailoStreamInterface = ConfigureParams = None
InputVStreamParams = OutputVStreamParams = FormatType = InferVStreams = None

HAILO_IMPORT_ERROR: Optional[BaseException] = None


def _load_hailo() -> None:
    """Bind the hailo_platform names on first use, or raise RuntimeError."""
    global HEF, VDevice, HailoStreamInterface, ConfigureParams
    global InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams
    global HAILO_IMPORT_ERROR

    if HEF is not None:
        return
    try:
        from hailo_platform import (
            HEF as _HEF,
            VDevice as _VDevice,
            HailoStreamInterface as _HailoStreamInterface,
            ConfigureParams as _ConfigureParams,
            InputVStreamParams as _InputVStreamParams,
            OutputVStreamParams as _OutputVStreamParams,
            FormatType as _FormatType,
            InferVStreams as _InferVStreams,
        )
    except Exception as exc:
        HAILO_IMPORT_ERROR = exc
        raise RuntimeError(
            f"Hailo runtime is not available ({exc}). Visual detection "
            f"cannot start; the acoustic subsystem is unaffected.") from exc

    HEF, VDevice = _HEF, _VDevice
    HailoStreamInterface, ConfigureParams = _HailoStreamInterface, _ConfigureParams
    InputVStreamParams, OutputVStreamParams = _InputVStreamParams, _OutputVStreamParams
    FormatType, InferVStreams = _FormatType, _InferVStreams


def hailo_available() -> bool:
    """True if the Hailo runtime can be imported. Used for graceful degradation."""
    try:
        _load_hailo()
        return True
    except RuntimeError:
        return False

HEAD_CONFIG: List[Tuple[str, str, int]] = [
    ("bestty_yolo26/conv61", "bestty_yolo26/conv64",  8),
    ("bestty_yolo26/conv77", "bestty_yolo26/conv80",  16),
    ("bestty_yolo26/conv91", "bestty_yolo26/conv94",  32),
]

DRONE_REAL_WIDTH_M: Optional[float] = 0.25
CAMERA_FOCAL_PX = {
   
    # SUSPECT — must be re-calibrated. This was measured at 5 m with a
    # 65.5 px box, i.e. by the same procedure that was just proven to
    # inflate the NEAR value 2x (see the note below). There is no second
    # data point for this camera and no datasheet to check against (the
    # IMX477 takes interchangeable lenses), so the true value cannot be
    # derived here — only measured. Re-run the 'c' calibration on this
    # camera at CALIBRATION_DISTANCE_M and replace this number.
    # MEASURED AND CONFIRMED — no longer suspect. Two independent
    # calibrations at very different distances agree:
    #     5.0 m ->  65.5 px -> focal 1310  (HFOV 27.5 deg)
    #     1.5 m -> 212.3 px -> focal 1274  (HFOV 28.2 deg)
    # For a pinhole camera (box x distance) must be constant: 327.5 vs
    # 318.5, only 2.8% apart. So this camera obeys the model and was
    # never wrong — the earlier suspicion that it was inflated like the
    # NEAR value is REFUTED by measurement. Using the 1.5 m figure, since
    # the larger box makes it the more precise of the two.
    CameraManager.FAR_CAMERA_ID:  1274.0,  # IMX477 + fitted lens, ~28 deg HFOV
  
    # CORRECTED. The old 1003.4 came from a 5 m calibration and was
    # proven wrong by two of your own measurements: at 5 m the box was
    # 50.2 px, at 1.5 m it was 83.6 px. For a pinhole camera
    # (box x distance) must be constant — those give 251 vs 125, exactly
    # 2.00x apart, so the 5 m point cannot be trusted. The 1.5 m point
    # implies 502 px = 65.1 deg HFOV, which matches the IMX708 Camera
    # Module 3 datasheet figure of 66 deg. The 5 m point implied 35.4
    # deg, narrower than the sensor physically is. Root cause: at 5 m a
    # 0.25 m drone should span only ~25 px, and detection boxes on
    # objects that small are systematically inflated.
    # UNVERIFIED — measure this properly (freeze with 'f', drone at
    # 1.5 m, press 'c') and replace the number.
    #
    # 501.7 was NOT measured. It was derived indirectly from a single
    # report ("shows 3 m when actually at 1.5 m"), on the theory that
    # detection boxes are inflated at long range. The FAR camera has
    # since disproved that theory: its 5 m and 1.5 m calibrations agree
    # to 2.8%, so the method is sound even with a small box. That leaves
    # this camera's two candidates unresolved:
    #     5 m calibration ->  50.2 px -> focal 1003 (HFOV 35.4 deg)
    #     derived 1.5 m   ->  83.6 px -> focal  502 (HFOV 65.1 deg)
    # The datasheet favours ~502 (IMX708 Camera Module 3 is ~66 deg),
    # the direct measurement favours ~1003. Only a real 1.5 m
    # calibration on this camera settles it — do not trust either until
    # then. If the measurement comes out near 1000, put 1003 back.
    CameraManager.NEAR_CAMERA_ID: 501.7,   # IMX708 — UNVERIFIED, re-measure
}


# Ground-truth distance for the 'c' calibration key.
#
# 1.5 m, NOT 5 m. This was corrected after 5 m was proven to produce a
# 2x-wrong focal length: the accuracy of this whole method depends on the
# detection box being TIGHT around the drone, and a box is only tight
# when it is large. At 5 m a 0.25 m drone spans ~25 px and the detector
# draws a visibly inflated box; at 1.5 m it spans ~85 px and the box is
# accurate (independently confirmed: the 1.5 m measurement reproduces the
# IMX708's published 66 deg field of view, the 5 m one does not).
#
# If you track a physically larger drone you can calibrate further away —
# what matters is the BOX SIZE IN PIXELS, not the distance itself. The
# guard below enforces this rather than trusting the distance.
CALIBRATION_DISTANCE_M = 1.5

# Minimum box width, in pixels, for a calibration to be accepted.
# Below this the box is too small for its edges to mean anything and the
# resulting focal length will be inflated — which is exactly the failure
# that produced the wrong NEAR value. 70 px is set from the measured
# evidence: the 83.6 px box gave a datasheet-correct result, the 50.2 px
# box was 2x wrong.
MIN_CALIBRATION_BOX_PX = 70.0


def focal_px_from_fov(fov_deg: float, image_dim_px: int) -> float:

    return (image_dim_px / 2.0) / float(np.tan(np.radians(fov_deg) / 2.0))


def estimate_distance_m(apparent_width_px: float,
                        camera_id: int) -> Optional[float]:

    if DRONE_REAL_WIDTH_M is None:
        return None
    focal_px = CAMERA_FOCAL_PX.get(camera_id)
    if not focal_px or apparent_width_px <= 1.0:
        return None
    return (DRONE_REAL_WIDTH_M * float(focal_px)) / float(apparent_width_px)

_MIN_SPEED_SQ_JAC: float = 1.0    # px²/s²  — minimum v² for phi/omega Jacobians
_MAX_VEL:          float = 500.0  # px/s
_MAX_ACCEL:        float = 300.0  # px/s²
_MAX_OMEGA:        float = 4.0    # rad/s
_EYE23 = np.eye(2, 3, dtype=np.float64)



def make_psd(P: np.ndarray, eps: float = 1e-11, max_eig: float = 1e8) -> np.ndarray:
    if not np.all(np.isfinite(P)):
        return np.eye(P.shape[0]) * eps
    P_sym = 0.5 * (P + P.T)
    try:
        _ = np.linalg.cholesky(P_sym)
        if np.max(np.diag(P_sym)) <= max_eig:
            return P_sym
    except np.linalg.LinAlgError:
        pass
    vals, vecs = np.linalg.eigh(P_sym)
    clipped = np.clip(vals, eps, max_eig)
    P_out = (vecs * clipped) @ vecs.T
    return 0.5 * (P_out + P_out.T)

def robust_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A_sym = 0.5 * (A + A.T)
    try:
        L = np.linalg.cholesky(A_sym)
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(A_sym)
        vals_inv = np.where(np.abs(vals) > 1e-11, 1.0 / vals, 0.0)
        return (vecs * vals_inv) @ vecs.T @ b


def robust_mahalanobis_sq(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    diff = x - mean
    sol  = robust_solve(cov, diff)
    return float(max(0.0, np.dot(diff, sol)))


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def safe_point(x, y, frame_shape=None) -> Optional[Tuple[int, int]]:
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(xf) and np.isfinite(yf)):
        return None
    xi, yi = int(round(xf)), int(round(yf))
    if frame_shape is not None:
        h, w = frame_shape[0], frame_shape[1]
        xi = int(np.clip(xi, -4 * w, 4 * w))
        yi = int(np.clip(yi, -4 * h, 4 * h))
    return xi, yi


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1 = max(x1, x2);        iy1 = max(y1, y2)
    ix2 = min(x1+w1, x2+w2);  iy2 = min(y1+h1, y2+h2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    union = w1*h1 + w2*h2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def get_jacobian_cv_to_ctrv(x_cv: np.ndarray) -> np.ndarray:
    vx, vy = float(x_cv[4]), float(x_cv[5])
    vx = np.clip(vx, -_MAX_VEL, _MAX_VEL)
    vy = np.clip(vy, -_MAX_VEL, _MAX_VEL)
    v_sq = vx*vx + vy*vy
    J = np.zeros((7, 6))
    J[0,0]=1.0; J[1,1]=1.0; J[2,2]=1.0; J[3,3]=1.0
    if v_sq >= _MIN_SPEED_SQ_JAC:
        v = np.sqrt(v_sq)
        J[4,4] =  vx / v;   J[4,5] =  vy / v
        J[5,4] = -vy / v_sq; J[5,5] =  vx / v_sq
    return J


def get_jacobian_ctrv_to_ca(x_ctrv: np.ndarray) -> np.ndarray:
    v     = float(np.clip(x_ctrv[4], 0.0, _MAX_VEL))
    phi   = float(x_ctrv[5])
    omega = float(np.clip(x_ctrv[6], -_MAX_OMEGA, _MAX_OMEGA))
    cp, sp = np.cos(phi), np.sin(phi)
    J = np.zeros((8, 7))
    J[0,0]=1.0; J[1,1]=1.0; J[2,2]=1.0; J[3,3]=1.0
    J[4,4]=cp;        J[4,5]=-v*sp
    J[5,4]=sp;        J[5,5]= v*cp
    J[6,4]=-omega*sp; J[6,5]=-v*omega*cp; J[6,6]=-v*sp
    J[7,4]= omega*cp; J[7,5]=-v*omega*sp; J[7,6]= v*cp
    return J


def get_jacobian_ca_to_cv(x_ca: np.ndarray) -> np.ndarray:
    J = np.zeros((6, 8))
    J[0,0]=1.0; J[1,1]=1.0; J[2,2]=1.0; J[3,3]=1.0
    J[4,4]=1.0; J[5,5]=1.0
    return J


def get_jacobian_cv_to_ca(x_cv: np.ndarray) -> np.ndarray:
    J = np.zeros((8, 6))
    J[0,0]=1.0; J[1,1]=1.0; J[2,2]=1.0; J[3,3]=1.0
    J[4,4]=1.0; J[5,5]=1.0
    return J


def get_jacobian_ca_to_ctrv(x_ca: np.ndarray) -> np.ndarray:
    vx = float(np.clip(x_ca[4], -_MAX_VEL, _MAX_VEL))
    vy = float(np.clip(x_ca[5], -_MAX_VEL, _MAX_VEL))
    ax = float(np.clip(x_ca[6], -_MAX_ACCEL, _MAX_ACCEL))
    ay = float(np.clip(x_ca[7], -_MAX_ACCEL, _MAX_ACCEL))
    v_sq = vx*vx + vy*vy
    J = np.zeros((7, 8))
    J[0,0]=1.0; J[1,1]=1.0; J[2,2]=1.0; J[3,3]=1.0
    if v_sq >= _MIN_SPEED_SQ_JAC:
        v         = np.sqrt(v_sq)
        v_sq_g    = v_sq        # already >= 1.0
        v_sq_g_sq = v_sq_g * v_sq_g
        omega_num = vx*ay - vy*ax
        J[4,4] =  vx / v;      J[4,5] =  vy / v
        J[5,4] = -vy / v_sq_g; J[5,5] =  vx / v_sq_g
        J[6,4] = (ay*v_sq_g  - 2.0*vx*omega_num) / v_sq_g_sq
        J[6,5] = (-ax*v_sq_g - 2.0*vy*omega_num) / v_sq_g_sq
        J[6,6] = -vy / v_sq_g
        J[6,7] =  vx / v_sq_g
    return J


def get_jacobian_ctrv_to_cv(x_ctrv: np.ndarray) -> np.ndarray:
    v   = float(np.clip(x_ctrv[4], 0.0, _MAX_VEL))
    phi = float(x_ctrv[5])
    cp, sp = np.cos(phi), np.sin(phi)
    J = np.zeros((6, 7))
    J[0,0]=1.0; J[1,1]=1.0; J[2,2]=1.0; J[3,3]=1.0
    J[4,4]=cp;  J[4,5]=-v*sp
    J[5,4]=sp;  J[5,5]= v*cp
    return J


class AppearanceFeatureExtractor:
    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim

    def extract(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x, y, w, h = map(int, map(round, bbox))
        ih, iw = frame_bgr.shape[:2]
        x1 = max(0, min(x, iw-1));  y1 = max(0, min(y, ih-1))
        x2 = max(0, min(x+w, iw));  y2 = max(0, min(y+h, ih))
        if (x2-x1) < 2 or (y2-y1) < 2:
            return np.zeros(self.feature_dim)
        crop = frame_bgr[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, [4,4,2],
                             [0,180, 0,256, 0,256])
        feat = hist.flatten()
        n = np.linalg.norm(feat)
        if n > 1e-12:
            feat /= n
        else:
            feat = np.zeros(self.feature_dim); feat[0] = 1.0
        return feat


class EgoMotionEstimator:
    def __init__(self):
        self.lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts:  Optional[np.ndarray] = None
        # PERF-5: re-detection backoff state (see estimate_motion).
        self._redetect_cooldown = 0
        self._redetect_failures = 0
        self._MAX_REDETECT_BACKOFF = 8

    def reset(self):

        self.prev_gray = None
        self.prev_pts = None
        self._redetect_cooldown = 0
        self._redetect_failures = 0

    def _detect_features(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            frame_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    def estimate_motion(self, frame_gray: np.ndarray) -> np.ndarray:
        warp = _EYE23.copy()
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            self.prev_pts  = self._detect_features(frame_gray)
            return warp

        if self.prev_pts is not None and len(self.prev_pts) > 10:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.prev_pts, None, **self.lk_params)
            if nxt is not None:
                good_p = self.prev_pts[st.flatten()==1]
                good_n = nxt[st.flatten()==1]
                if len(good_p) > 8:
                    M, _ = cv2.estimateAffinePartial2D(good_p, good_n, method=cv2.RANSAC)
                    if M is not None:
                        warp = M
                self.prev_pts = good_n.reshape(-1, 1, 2)
            else:
                self.prev_pts = None

        if self.prev_pts is None or len(self.prev_pts) < 30:

            if self._redetect_cooldown > 0:
                self._redetect_cooldown -= 1
            else:
                self.prev_pts = self._detect_features(frame_gray)
                found = 0 if self.prev_pts is None else len(self.prev_pts)
                if found <= 10:
                    self._redetect_failures = min(self._redetect_failures + 1, 3)
                    self._redetect_cooldown = min(
                        2 ** self._redetect_failures, self._MAX_REDETECT_BACKOFF)
                else:
                    self._redetect_failures = 0
                    self._redetect_cooldown = 0

        self.prev_gray = frame_gray
        return warp


class TrackState:
    Tentative = "Tentative"
    Confirmed = "Confirmed"
    Lost      = "Lost"
    Deleted   = "Deleted"


class ModelType:
    CV   = "CV"
    CTRV = "CTRV"
    CA   = "CA"

def build_initial_covariance(state_dim: int, model_type: str) -> np.ndarray:
    if model_type == ModelType.CV:
        diag = [25.0, 25.0, 25.0, 25.0, 22500.0, 22500.0]
    elif model_type == ModelType.CTRV:
        diag = [25.0, 25.0, 25.0, 25.0, 22500.0, float(np.pi**2), 4.0]
    elif model_type == ModelType.CA:
        diag = [25.0, 25.0, 25.0, 25.0, 22500.0, 22500.0, 2500.0, 2500.0]
    else:
        diag = [1.0] * state_dim
    return np.diag(diag)


class StableUKF:
    def __init__(self, state_dim: int, model_type: str):
        self.state_dim  = state_dim
        self.model_type = model_type
        self.x = np.zeros(state_dim)
        self.P = build_initial_covariance(state_dim, model_type)
        self.alpha  = 1e-2
        self.beta   = 2.0
        self.kappa  = 0.0
        self._compute_weights()

    def _compute_weights(self):
        n = self.state_dim
        self.lam = self.alpha**2 * (n + self.kappa) - n
        self.w_m = np.full(2*n+1, 0.5/(n+self.lam))
        self.w_c = np.full(2*n+1, 0.5/(n+self.lam))
        self.w_m[0] = self.lam / (n + self.lam)
        self.w_c[0] = self.w_m[0] + (1.0 - self.alpha**2 + self.beta)

    def _clamp_state(self):
        """Clamp state to physical bounds to prevent Q/Jacobian overflow."""
        if self.model_type == ModelType.CV:
            self.x[4] = float(np.clip(self.x[4], -_MAX_VEL, _MAX_VEL))
            self.x[5] = float(np.clip(self.x[5], -_MAX_VEL, _MAX_VEL))
        elif self.model_type == ModelType.CTRV:
            self.x[4] = float(np.clip(self.x[4], 0.0, _MAX_VEL))
            self.x[5] = wrap_angle(self.x[5])
            self.x[6] = float(np.clip(self.x[6], -_MAX_OMEGA, _MAX_OMEGA))
        elif self.model_type == ModelType.CA:
            self.x[4] = float(np.clip(self.x[4], -_MAX_VEL, _MAX_VEL))
            self.x[5] = float(np.clip(self.x[5], -_MAX_VEL, _MAX_VEL))
            self.x[6] = float(np.clip(self.x[6], -_MAX_ACCEL, _MAX_ACCEL))
            self.x[7] = float(np.clip(self.x[7], -_MAX_ACCEL, _MAX_ACCEL))

    def _generate_sigma_points(self) -> np.ndarray:
        n = self.state_dim
        self.P = make_psd(self.P)
        try:
            L = np.linalg.cholesky((n + self.lam) * self.P)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh((n + self.lam) * self.P)
            L = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-11)))
        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = self.x
        for i in range(n):
            sigmas[i+1]   = self.x + L[:, i]
            sigmas[i+1+n] = self.x - L[:, i]
        if self.model_type == ModelType.CTRV:
            sigmas[:, 5] = np.vectorize(wrap_angle)(sigmas[:, 5])
        return sigmas

    def _predict_dynamics(self, sp: np.ndarray, dt: float) -> np.ndarray:
        out = np.zeros_like(sp)
        if self.model_type == ModelType.CV:
            cx, cy, w, h, vx, vy = sp.T
            out[:, 0] = cx + vx*dt;  out[:, 1] = cy + vy*dt
            out[:, 2] = w;            out[:, 3] = h
            out[:, 4] = vx;           out[:, 5] = vy
        elif self.model_type == ModelType.CTRV:
            cx, cy, w, h, v, phi, omega = sp.T
            sm = np.abs(omega) <= 1e-5
            lg = ~sm
            cpx = np.zeros_like(cx); cpy = np.zeros_like(cy)
            if np.any(lg):
                cpx[lg] = cx[lg] + (v[lg]/omega[lg])*(np.sin(phi[lg]+omega[lg]*dt)-np.sin(phi[lg]))
                cpy[lg] = cy[lg] + (v[lg]/omega[lg])*(-np.cos(phi[lg]+omega[lg]*dt)+np.cos(phi[lg]))
            if np.any(sm):
                cpx[sm] = cx[sm] + v[sm]*np.cos(phi[sm])*dt
                cpy[sm] = cy[sm] + v[sm]*np.sin(phi[sm])*dt
            out[:, 0] = cpx;  out[:, 1] = cpy
            out[:, 2] = w;    out[:, 3] = h
            out[:, 4] = v
            out[:, 5] = np.vectorize(wrap_angle)(phi + omega*dt)
            out[:, 6] = omega
        elif self.model_type == ModelType.CA:
            cx, cy, w, h, vx, vy, ax, ay = sp.T
            out[:, 0] = cx + vx*dt + 0.5*ax*dt*dt
            out[:, 1] = cy + vy*dt + 0.5*ay*dt*dt
            out[:, 2] = w;   out[:, 3] = h
            out[:, 4] = vx + ax*dt;  out[:, 5] = vy + ay*dt
            out[:, 6] = ax;          out[:, 7] = ay
        return out

    def _compute_Q(self, dt: float) -> np.ndarray:
        if self.model_type == ModelType.CV:
            vx, vy = self.x[4], self.x[5]
            spd   = min(np.sqrt(vx*vx + vy*vy), _MAX_VEL)  # clamp prevents Q explosion
            sa    = 0.5*(1.0 + 0.05*spd)
            Q = np.zeros((6, 6))
            Q[0,0]=(dt**3)/3*sa**2; Q[0,4]=(dt**2)/2*sa**2; Q[4,0]=Q[0,4]; Q[4,4]=dt*sa**2
            Q[1,1]=(dt**3)/3*sa**2; Q[1,5]=(dt**2)/2*sa**2; Q[5,1]=Q[1,5]; Q[5,5]=dt*sa**2
            Q[2,2]=dt*0.0025; Q[3,3]=dt*0.0025
            return Q
        elif self.model_type == ModelType.CTRV:
            v   = float(np.clip(self.x[4], 0.0, _MAX_VEL))
            sa  = 0.5*(1.0+0.05*v); svd = sa; spd = 0.1; sod = 0.05
            Q = np.zeros((7, 7))
            Q[0,0]=(dt**3)/3*sa**2; Q[1,1]=(dt**3)/3*sa**2
            Q[2,2]=dt*0.0025;        Q[3,3]=dt*0.0025
            Q[4,4]=dt*svd**2;        Q[5,5]=dt*spd**2; Q[6,6]=dt*sod**2
            return Q
        elif self.model_type == ModelType.CA:
            ax, ay = self.x[6], self.x[7]
            acc    = min(np.sqrt(ax*ax + ay*ay), _MAX_ACCEL)  # clamp
            sj     = 0.5*(1.0+0.1*acc)
            Q = np.zeros((8, 8))
            Q[0,0]=(dt**5)/20*sj**2; Q[0,4]=(dt**4)/8*sj**2; Q[0,6]=(dt**3)/6*sj**2
            Q[4,0]=Q[0,4]; Q[4,4]=(dt**3)/3*sj**2; Q[4,6]=(dt**2)/2*sj**2
            Q[6,0]=Q[0,6]; Q[6,4]=Q[4,6]; Q[6,6]=dt*sj**2
            Q[1,1]=(dt**5)/20*sj**2; Q[1,5]=(dt**4)/8*sj**2; Q[1,7]=(dt**3)/6*sj**2
            Q[5,1]=Q[1,5]; Q[5,5]=(dt**3)/3*sj**2; Q[5,7]=(dt**2)/2*sj**2
            Q[7,1]=Q[1,7]; Q[7,5]=Q[5,7]; Q[7,7]=dt*sj**2
            Q[2,2]=dt*0.0025; Q[3,3]=dt*0.0025
            return Q
        return np.eye(self.state_dim) * 0.1

    def predict(self, dt: float):
        Q   = self._compute_Q(dt)
        sp  = self._generate_sigma_points()
        spf = self._predict_dynamics(sp, dt)

        if self.model_type == ModelType.CTRV:
            s_phi = np.sum(self.w_m * np.sin(spf[:, 5]))
            c_phi = np.sum(self.w_m * np.cos(spf[:, 5]))
            x_pred = np.einsum('ij,i->j', spf, self.w_m)
            x_pred[5] = wrap_angle(np.arctan2(s_phi, c_phi))
        else:
            x_pred = np.einsum('ij,i->j', spf, self.w_m)

        diff = spf - x_pred
        if self.model_type == ModelType.CTRV:
            diff[:, 5] = np.vectorize(wrap_angle)(diff[:, 5])
        P_pred = np.einsum('ij,i,ik->jk', diff, self.w_c, diff)

        self.x = x_pred
        self.P = make_psd(P_pred + Q)
        self._clamp_state()

    def update(self, z: np.ndarray, R: np.ndarray) -> float:
        sp    = self._generate_sigma_points()
        z_sp  = sp[:, 0:4]
        z_hat = np.einsum('ij,i->j', z_sp, self.w_m)
        zd    = z_sp - z_hat
        xd    = sp - self.x
        if self.model_type == ModelType.CTRV:
            xd[:, 5] = np.vectorize(wrap_angle)(xd[:, 5])

        S  = np.einsum('ij,i,ik->jk', zd, self.w_c, zd) + R
        Tc = np.einsum('ij,i,ik->jk', xd, self.w_c, zd)
        K  = robust_solve(S, Tc.T).T

        innov  = z - z_hat
        self.x = self.x + K @ innov
        if self.model_type == ModelType.CTRV:
            self.x[5] = wrap_angle(self.x[5])
        self.P = make_psd(self.P - K @ S @ K.T)
        self._clamp_state()

        # Log-likelihood for IMM weight update
        dim = len(z)
        try:
            Ls       = np.linalg.cholesky(S)
            log_detS = 2.0 * np.sum(np.log(np.diag(Ls)))
            y        = np.linalg.solve(Ls, innov)
            mah      = float(np.dot(y, y))
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(0.5*(S+S.T))
            vals       = np.maximum(vals, 1e-11)
            log_detS   = np.sum(np.log(vals))
            mah        = float(innov @ (vecs*(1.0/vals)) @ vecs.T @ innov)
        return -0.5*(dim*np.log(2.0*np.pi) + log_detS + mah)


class IMMEstimator:
    def __init__(self):
        self.filters = {
            ModelType.CV:   StableUKF(6, ModelType.CV),
            ModelType.CTRV: StableUKF(7, ModelType.CTRV),
            ModelType.CA:   StableUKF(8, ModelType.CA),
        }
        self.mu         = np.full(3, 1.0/3.0)
        self.model_list = [ModelType.CV, ModelType.CTRV, ModelType.CA]
        self.p_ij       = np.array([[0.85, 0.10, 0.05],
                                     [0.10, 0.80, 0.10],
                                     [0.05, 0.15, 0.80]])
        self.mu_prev_mix = self.mu.copy()

    def warp_all_models(self, A: np.ndarray):
        # FIX: skip when near-identity (ego-motion ≈ 0)
        if np.allclose(A, _EYE23, atol=1e-4):
            return
        R_w   = A[:, 0:2]
        T_w   = A[:, 2]
        theta = np.arctan2(R_w[1, 0], R_w[0, 0])

        cv = self.filters[ModelType.CV]
        cv.x[0:2] = R_w @ cv.x[0:2] + T_w
        cv.x[4:6] = R_w @ cv.x[4:6]
        J = np.eye(6); J[0:2,0:2]=R_w; J[4:6,4:6]=R_w
        cv.P = make_psd(J @ cv.P @ J.T)

        ctrv = self.filters[ModelType.CTRV]
        ctrv.x[0:2] = R_w @ ctrv.x[0:2] + T_w
        ctrv.x[5]   = wrap_angle(ctrv.x[5] + theta)
        J = np.eye(7); J[0:2,0:2]=R_w
        ctrv.P = make_psd(J @ ctrv.P @ J.T)

        ca = self.filters[ModelType.CA]
        ca.x[0:2] = R_w @ ca.x[0:2] + T_w
        ca.x[4:6] = R_w @ ca.x[4:6]
        ca.x[6:8] = R_w @ ca.x[6:8]
        J = np.eye(8); J[0:2,0:2]=R_w; J[4:6,4:6]=R_w; J[6:8,6:8]=R_w
        ca.P = make_psd(J @ ca.P @ J.T)

    def mix_states(self):
        c_j = np.maximum(self.mu @ self.p_ij, 1e-15)
        mu  = self.mu
        mix = np.outer(mu, np.ones(3)) * self.p_ij / c_j  # (3,3) mix_coeff[i,j]

        mixed_x = {m: np.zeros(self.filters[m].state_dim) for m in self.model_list}
        mixed_P = {m: np.zeros((self.filters[m].state_dim,
                                 self.filters[m].state_dim)) for m in self.model_list}

        for j, tm in enumerate(self.model_list):
            dim_j = self.filters[tm].state_dim
            mx    = np.zeros(dim_j)

            if tm == ModelType.CTRV:
                ss = cs = 0.0
                for i, sm in enumerate(self.model_list):
                    xp = self._project_state(self.filters[sm].x, sm, tm)
                    w  = mix[i, j]
                    ss += w * np.sin(xp[5])
                    cs += w * np.cos(xp[5])
                    for k in range(dim_j):
                        if k != 5:
                            mx[k] += w * xp[k]
                mx[5] = wrap_angle(np.arctan2(ss, cs))
            else:
                for i, sm in enumerate(self.model_list):
                    mx += mix[i, j] * self._project_state(
                        self.filters[sm].x, sm, tm)
            mixed_x[tm] = mx

        for j, tm in enumerate(self.model_list):
            dim_j = self.filters[tm].state_dim
            mP    = np.zeros((dim_j, dim_j))
            mx    = mixed_x[tm]
            for i, sm in enumerate(self.model_list):
                xs  = self.filters[sm].x
                xp  = self._project_state(xs, sm, tm)
                Ft  = self._get_transition_jacobian(xs, sm, tm)
                Pp  = Ft @ self.filters[sm].P @ Ft.T

                # Extra uncertainty for undefined angular states
                if sm in (ModelType.CV, ModelType.CA) and tm == ModelType.CTRV:
                    v_sq_src = xs[4]**2 + xs[5]**2
                    if v_sq_src < _MIN_SPEED_SQ_JAC:
                        # phi and omega are undefined at near-zero speed.
                        # Assign maximum-entropy uncertainty independently.
                        Pp[5, :] = 0.0; Pp[:, 5] = 0.0
                        Pp[5, 5] = float(np.pi**2)
                        Pp[6, :] = 0.0; Pp[:, 6] = 0.0
                        Pp[6, 6] = 4.0
                elif sm == ModelType.CV and tm == ModelType.CA:
                    Pp[6, 6] += 0.1; Pp[7, 7] += 0.1

                diff    = xp - mx
                if tm == ModelType.CTRV:
                    diff[5] = wrap_angle(diff[5])
                mP += mix[i, j] * (Pp + np.outer(diff, diff))

            mixed_P[tm] = make_psd(mP)

        for m in self.model_list:
            self.filters[m].x = mixed_x[m]
            self.filters[m].P = mixed_P[m]
        self.mu_prev_mix = c_j

    def predict(self, dt: float):
        self.mix_states()
        for m in self.model_list:
            self.filters[m].predict(dt)

    def update(self, z: np.ndarray, R: np.ndarray) -> np.ndarray:
        ll = np.array([self.filters[m].update(z, R) for m in self.model_list])
        lj = ll + np.log(np.maximum(self.mu_prev_mix, 1e-15))
        ex = np.exp(lj - np.max(lj))
        self.mu = ex / max(ex.sum(), 1e-15)
        self.mu = np.maximum(self.mu, 1e-4)
        self.mu /= self.mu.sum()
        return self.mu

    def get_merged_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        mx = np.zeros(8)
        for i, m in enumerate(self.model_list):
            mx += self.mu[i] * self._project_state(self.filters[m].x, m, ModelType.CA)
        mP = np.zeros((8, 8))
        for i, m in enumerate(self.model_list):
            xca = self._project_state(self.filters[m].x, m, ModelType.CA)
            Ft  = self._get_transition_jacobian(self.filters[m].x, m, ModelType.CA)
            Pca = Ft @ self.filters[m].P @ Ft.T
            if m == ModelType.CV:
                Pca[6,6] += 0.1; Pca[7,7] += 0.1
            d   = xca - mx
            mP += self.mu[i] * (Pca + np.outer(d, d))
        return mx, make_psd(mP)

    def _project_state(self, x: np.ndarray, fm: str, tm: str) -> np.ndarray:
        if fm == tm:
            return x.copy()
        if fm == ModelType.CV:
            cx,cy,w,h,vx,vy = x
            if tm == ModelType.CTRV:
                v   = np.sqrt(vx*vx + vy*vy)
                phi = np.arctan2(vy, vx)
                return np.array([cx,cy,w,h,v,phi,0.0])
            if tm == ModelType.CA:
                return np.array([cx,cy,w,h,vx,vy,0.0,0.0])
        elif fm == ModelType.CTRV:
            cx,cy,w,h,v,phi,omega = x
            vx=v*np.cos(phi); vy=v*np.sin(phi)
            if tm == ModelType.CV:
                return np.array([cx,cy,w,h,vx,vy])
            if tm == ModelType.CA:
                ax=-v*omega*np.sin(phi); ay=v*omega*np.cos(phi)
                return np.array([cx,cy,w,h,vx,vy,ax,ay])
        elif fm == ModelType.CA:
            cx,cy,w,h,vx,vy,ax,ay = x
            if tm == ModelType.CV:
                return np.array([cx,cy,w,h,vx,vy])
            if tm == ModelType.CTRV:
                v    = np.sqrt(vx*vx + vy*vy)
                phi  = np.arctan2(vy, vx)
                vsq  = vx*vx + vy*vy
                omega= (vx*ay-vy*ax)/vsq if vsq > _MIN_SPEED_SQ_JAC else 0.0
                return np.array([cx,cy,w,h,v,phi,omega])
        return x.copy()

    def _get_transition_jacobian(self, x: np.ndarray, fm: str, tm: str) -> np.ndarray:
        if fm == tm:
            return np.eye(self.filters[fm].state_dim)
        if fm == ModelType.CV:
            if tm == ModelType.CTRV: return get_jacobian_cv_to_ctrv(x)
            if tm == ModelType.CA:   return get_jacobian_cv_to_ca(x)
        elif fm == ModelType.CTRV:
            if tm == ModelType.CV:   return get_jacobian_ctrv_to_cv(x)
            if tm == ModelType.CA:   return get_jacobian_ctrv_to_ca(x)
        elif fm == ModelType.CA:
            if tm == ModelType.CV:   return get_jacobian_ca_to_cv(x)
            if tm == ModelType.CTRV: return get_jacobian_ca_to_ctrv(x)
        return np.eye(self.filters[tm].state_dim, self.filters[fm].state_dim)

    def clone(self) -> 'IMMEstimator':
        c = IMMEstimator()
        c.mu = self.mu.copy(); c.mu_prev_mix = self.mu_prev_mix.copy()
        for m in self.model_list:
            c.filters[m].x = self.filters[m].x.copy()
            c.filters[m].P = self.filters[m].P.copy()
        return c


class AppearanceModel:
    def __init__(self, alpha: float = 0.95):
        self.alpha    = alpha
        self.feature: Optional[np.ndarray] = None
        self.gallery: List[np.ndarray]     = []
        self.max_gallery_size              = 5

    def update(self, feat: np.ndarray):
        n = np.linalg.norm(feat)
        u = feat / (n if n > 1e-12 else 1.0)
        if self.feature is None:
            self.feature = u.copy()
        else:
            self.feature = self.alpha*self.feature + (1.0-self.alpha)*u
            self.feature /= np.linalg.norm(self.feature)
        self.gallery.append(u.copy())
        if len(self.gallery) > self.max_gallery_size:
            self.gallery.pop(0)

    def match_distance(self, feat: np.ndarray) -> float:
        if self.feature is None:
            return 1.0
        n = np.linalg.norm(feat)
        u = feat / (n if n > 1e-12 else 1.0)
        d = [1.0 - float(np.dot(self.feature, u))]
        d += [1.0 - float(np.dot(g, u)) for g in self.gallery]
        return min(d)


class TrajectoryHistory:
    def __init__(self):
        self.states:       List[np.ndarray] = []
        self.covariances:  List[np.ndarray] = []
        self.measurements: List[np.ndarray] = []
        self.mus:          List[np.ndarray] = []

    def record(self, state: np.ndarray, cov: np.ndarray,
               z: Optional[np.ndarray], mu: np.ndarray):
        self.states.append(state.copy())
        self.covariances.append(cov.copy())
        self.mus.append(mu.copy())
        self.measurements.append(
            z.copy() if z is not None
            else np.array([state[0], state[1], state[2], state[3]]))


class ADAsTrack:
    def __init__(self, track_id: int, initial_z: np.ndarray,
                 initial_feat: np.ndarray, min_hits: int = 3, max_age: int = 10,
                 detection_score: Optional[float] = None):
        self.id   = track_id
        self.imm  = IMMEstimator()
        for m in self.imm.model_list:
            self.imm.filters[m].x[0:4] = initial_z
        self.age                = 1
        self.hits               = 1
        self.consecutive_misses = 0
        self.state              = TrackState.Tentative
        self.min_hits           = min_hits
        self.max_age            = max_age
        # tentative_miss_tolerance=3: proof in analysis_results.md
        self.tentative_miss_tolerance = 3
        self.confirmed_miss_tolerance = 3
        self.quality_score = 0.5
        self.r_scale       = 1.0
        # INTEGRATION: real detector confidence of the most recent detection
        # matched to this track. None while the track is coasting on
        # prediction alone (no detection this frame) — the unified UI shows
        # that as a stale/absent value rather than reusing an old number as
        # if it were current.
        self.last_detection_score: Optional[float] = (
            None if detection_score is None else float(detection_score))
        self.last_detection_time: Optional[float] = (
            None if detection_score is None else time.monotonic())
        self.appearance    = AppearanceModel(alpha=0.95)
        self.appearance.update(initial_feat)
        self.history       = TrajectoryHistory()
        mx, mP = self.imm.get_merged_estimate()
        # PERF follow-up: cache the merged estimate on the track. It used
        # to be recomputed from scratch (3x Jacobian matmul + PSD repair
        # per call) up to 5 times per track per frame across
        # _prune_divergent (x2), _associate's gating loop, and main()'s
        # draw loop, even though nothing mutates self.imm between most of
        # those calls within one frame — traced every call site (grep
        # "get_merged_estimate" — 6 total) to confirm this. Only
        # predict()/update()/mark_missed() actually change self.imm's
        # state, so those are the only 3 places that need to recompute;
        # everywhere else now reads self.merged_x/self.merged_P.
        self.merged_x, self.merged_P = mx, mP
        self.history.record(mx, mP, initial_z, self.imm.mu)

    def predict(self, dt: float):
        self.imm.predict(dt)
        self.age += 1
        self.merged_x, self.merged_P = self.imm.get_merged_estimate()

    def update(self, z: np.ndarray, feat: np.ndarray, R_base: np.ndarray,
               detection_score: Optional[float] = None):
        # detection_score defaults to None so every pre-existing call site
        # keeps working unchanged.
        if detection_score is not None:
            self.last_detection_score = float(detection_score)
            self.last_detection_time  = time.monotonic()

        adaptive_R = R_base * self.r_scale
        self.imm.update(z, adaptive_R)
        self.appearance.update(feat)
        self.hits              += 1
        self.consecutive_misses = 0

        if self.state in (TrackState.Tentative, TrackState.Lost) \
                and self.hits >= self.min_hits:
            self.state = TrackState.Confirmed

        mx, mP   = self.imm.get_merged_estimate()
        self.merged_x, self.merged_P = mx, mP
        z_pred   = mx[0:4]
        S        = mP[0:4, 0:4] + adaptive_R
        nis      = robust_mahalanobis_sq(z, z_pred, S)

        # FIX-H2: clamp scale_val before EMA to prevent first-frame r_scale explosion
        scale_val  = float(np.clip(nis / 4.0, 0.1, 5.0))
        self.r_scale = float(np.clip(0.95*self.r_scale + 0.05*scale_val, 0.5, 5.0))

        self.history.record(mx, mP, z, self.imm.mu)
        tr = np.trace(mP[0:2, 0:2])
        self.quality_score = 0.7*self.quality_score + 0.3*np.exp(-tr/5.0)

    def mark_missed(self):
        self.consecutive_misses += 1
        mx, mP = self.imm.get_merged_estimate()
        self.merged_x, self.merged_P = mx, mP
        self.history.record(mx, mP, None, self.imm.mu)
        self.quality_score = max(0.0, self.quality_score - 0.15)

        if self.state == TrackState.Tentative:
            if self.consecutive_misses > self.tentative_miss_tolerance:
                self.state = TrackState.Deleted
        elif self.state == TrackState.Confirmed:
            if self.consecutive_misses > self.confirmed_miss_tolerance:
                self.state = TrackState.Lost
        elif self.state == TrackState.Lost:
            if self.consecutive_misses > self.max_age:
                self.state = TrackState.Deleted

class AdvancedADASTracker:
    def __init__(self):
        self.tracks:             List[ADAsTrack] = []
        self.track_id_counter    = 1
        self.ego_estimator       = EgoMotionEstimator()
        self.gating_threshold    = 16.27
        self.lambda_iou          = 0.2
        self.lambda_reid         = 0.3
        self.w_mahalanobis       = 0.5
        self.feature_extractor   = AppearanceFeatureExtractor(feature_dim=32)
        self.min_hits            = 3
        self.max_age             = 10
        self.matching_threshold  = 1.2
        self.last_update_time:   Optional[float] = None
        self.min_dt              = 0.01
        self.max_dt              = 0.5

    def _compute_dt(self) -> float:
        # AUDIT BUG #5: time.monotonic() used instead of time.time() (wall
        # clock) so interval measurement is immune to system clock jumps.
        now = time.monotonic()
        dt  = 0.1 if self.last_update_time is None else float(
            np.clip(now - self.last_update_time, self.min_dt, self.max_dt))
        self.last_update_time = now
        return dt

    def process_frame(self, frame_bgr: np.ndarray, detections: List[np.ndarray],
                       run_association: bool = True,
                       detection_scores: Optional[List[float]] = None
                       ) -> List['ADAsTrack']:
        # detection_scores is optional and defaults to None, so the original
        # 3-argument call signature behaves exactly as before.
        dt = self._compute_dt()

        if self.tracks:
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            warp       = self.ego_estimator.estimate_motion(frame_gray)

            for t in self.tracks:
                t.imm.warp_all_models(warp)   # skips when ≈ identity
                t.predict(dt)
        else:
            self.ego_estimator.reset()

        self.tracks = self._prune_divergent(self.tracks)

        if run_association:
            feats = [self.feature_extractor.extract(frame_bgr, d) for d in detections]
            self._associate(detections, feats, detection_scores)

        self.tracks = self._prune_divergent(self.tracks)
        self.tracks = [t for t in self.tracks if t.state != TrackState.Deleted]
        return [t for t in self.tracks if t.state == TrackState.Confirmed]

    @staticmethod
    def _prune_divergent(tracks: List['ADAsTrack']) -> List['ADAsTrack']:
        # PERF follow-up: read the cached merged estimate (kept fresh by
        # predict()/update()/mark_missed()) instead of recomputing it —
        # see the caching note in ADAsTrack.__init__.
        out = []
        for t in tracks:
            mx, mP = t.merged_x, t.merged_P
            if np.all(np.isfinite(mx)) and np.all(np.isfinite(mP)):
                out.append(t)
        return out

    def _associate(self, detections: List[np.ndarray], feats: List[np.ndarray],
                   scores: Optional[List[float]] = None):
        z_dets = [np.array([d[0]+d[2]/2.0, d[1]+d[3]/2.0, d[2], d[3]])
                  for d in detections]

        def score_of(i: int) -> Optional[float]:
            """Detector confidence of detection i, or None if not supplied."""
            if scores is None or i >= len(scores):
                return None
            return float(scores[i])

        if not self.tracks:
            for i, (zd, f) in enumerate(zip(z_dets, feats)):
                self.tracks.append(
                    ADAsTrack(self.track_id_counter, zd, f, self.min_hits,
                              self.max_age, score_of(i)))
                self.track_id_counter += 1
            return

        nt, nd = len(self.tracks), len(detections)
        C      = np.ones((nt, nd)) * 1e5

        # AUDIT BUG #10: removed a dead, independently-incorrect duplicate
        # normalization block that used to precede this one (sequential
        # in-place division mutated ws before wi/wr divided by it, and its
        # result was never read). This block was already the one actually
        # used.
        _s = self.w_mahalanobis; _i = self.lambda_iou; _r = self.lambda_reid
        _tot = _s+_i+_r; wm=_s/_tot; wi=_i/_tot; wr=_r/_tot

        r_base_list: List[Optional[np.ndarray]] = [None]*nt

        for ti, t in enumerate(self.tracks):
            # PERF follow-up: cached value — see ADAsTrack.__init__. Safe
            # here because this gating loop runs before any update()/
            # mark_missed() call in this same _associate() invocation, so
            # the cache still holds the post-predict() (pre-association)
            # estimate, which is exactly what gating should use.
            mx, mP   = t.merged_x, t.merged_P
            pz       = mx[0:4]
            pc       = mP[0:4, 0:4]
            pbox     = np.array([pz[0]-pz[2]/2, pz[1]-pz[3]/2, pz[2], pz[3]])
            area     = max(100.0, pz[2]*pz[3])
            sa       = float(np.clip(np.sqrt(area)/20.0, 0.5, 2.0))
            Rb       = np.diag([2.0*sa, 2.0*sa, 4.0*sa, 4.0*sa])
            r_base_list[ti] = Rb
            S        = pc + Rb*t.r_scale

            for di, det in enumerate(detections):
                zd     = z_dets[di]
                md     = robust_mahalanobis_sq(zd, pz, S)
                if md < self.gating_threshold:
                    sc = md / self.gating_threshold
                    ic = 1.0 - compute_iou(pbox, det)
                    rc = float(np.clip(t.appearance.match_distance(feats[di]), 0.0, 1.0))
                    C[ti, di] = wm*sc + wi*ic + wr*rc

        row_ind, col_ind = linear_sum_assignment(C)
        at, ad = set(), set()
        for r, c in zip(row_ind, col_ind):
            if C[r, c] < self.matching_threshold:
                self.tracks[r].update(z_dets[c], feats[c], r_base_list[r],
                                      score_of(c))
                at.add(r); ad.add(c)

        for r in range(nt):
            if r not in at:
                self.tracks[r].mark_missed()

        for c in range(nd):
            if c not in ad:
                self.tracks.append(
                    ADAsTrack(self.track_id_counter, z_dets[c], feats[c],
                               self.min_hits, self.max_age, score_of(c)))
                self.track_id_counter += 1

class HailoInference:
    def __init__(self, hef_path: str,
                 conf_thresh: float = 0.4,
                 iou_thresh:  float = 0.45,
                 input_shape: Tuple[int, int] = (640, 640)):
        _load_hailo()

        self.conf_thresh  = conf_thresh
        self.iou_thresh   = iou_thresh
        self.input_shape  = input_shape   # (W, H)

        self.hef    = HEF(hef_path)
        self.target = VDevice()
        cfg_params  = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group        = self.target.configure(self.hef, cfg_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstream_info  = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        self.input_params  = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32)

        self._act_ctx = self.network_group.activate(self.network_group_params)
        self._act_ctx.__enter__()
        self._vstreams = InferVStreams(
            self.network_group, self.input_params, self.output_params)
        self._pipeline = self._vstreams.__enter__()

        self._in_buf = np.full(
            (1, input_shape[1], input_shape[0], 3), 114, dtype=np.uint8)
        self._lb_buf = self._in_buf[0]
        self._cls_needs_sigmoid: Optional[bool] = None
        self._cls_logit_min   = float('inf')
        self._cls_logit_max   = float('-inf')
        self._cls_calls_seen  = 0
        self._CLS_MIN_CALLS   = 3

        self._ltrb_mode: Optional[str] = None   # 'stride' or 'pixel'
        self._ltrb_stride_score = 0.0
        self._ltrb_pixel_score  = 0.0
        self._ltrb_samples_seen = 0
        self._LTRB_MIN_SAMPLES  = 30

        # Verify HEAD_CONFIG against HEF output names
        self._hef_output_names = {info.name for info in self.output_vstream_info}
        self._head_config_valid = self._verify_head_config()

    def _verify_head_config(self) -> bool:
        missing = [n for bbox, cls, _ in HEAD_CONFIG
                   for n in (bbox, cls) if n not in self._hef_output_names]
        if missing:
            print("[HailoInference] HEAD_CONFIG has names NOT found in HEF:")
            for n in missing:
                print(f"  MISSING: {n}")
            print("[HailoInference] Available HEF output tensors:")
            for n in sorted(self._hef_output_names):
                print(f"  {n}")
            print("[HailoInference] Falling back to shape-based auto-detection.")
            return False
        print("[HailoInference] HEAD_CONFIG verified against HEF. All names match.")
        return True

    @staticmethod
    def _to_hwc(arr: np.ndarray) -> np.ndarray:
        """Ensure tensor is HWC. Converts CHW if first dim is 1 or 4."""
        if arr.ndim == 2:
            return arr[:, :, np.newaxis]
        if arr.ndim == 3:
            c, h, w = arr.shape
            if c in (1, 4) and h == w and c != h:
                return np.transpose(arr, (1, 2, 0))
        return arr

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Letterbox into pre-allocated buffer. Returns (buf, ratio, pad_w, pad_h)."""
        sh, sw = img.shape[:2]
        dw, dh = self.input_shape
        r      = min(dh/sh, dw/sw)
        nw, nh = int(round(sw*r)), int(round(sh*r))
        pw     = (dw - nw) / 2.0
        ph     = (dh - nh) / 2.0
        l      = int(round(pw - 0.1))
        t      = int(round(ph - 0.1))
        self._lb_buf[:] = 114
        if nw == sw and nh == sh:
            # PERF follow-up: ratio == 1.0 exactly (e.g. this deployment's
            # 640x480 camera into a 640x640 model input, which is pure
            # padding — see TWO_CAMERAS_AUDIT.md Section 7). Skip
            # cv2.resize entirely: resizing to the identical size does the
            # same work as a plain copy but slower, since it still runs a
            # full interpolation pass over every pixel.
            self._lb_buf[t:t+nh, l:l+nw] = img
        else:
            self._lb_buf[t:t+nh, l:l+nw] = cv2.resize(
                img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # AUDIT BUG #7: return the actual integer offsets (l, t) used to
        # place the pixels, not the unrounded float (pw, ph) — the inverse
        # transform in _decode_head must undo the same offset that was
        # applied here, or every decoded box carries a sub-pixel error.
        return self._lb_buf, r, float(l), float(t)

    def _update_ltrb_vote(self, regs: np.ndarray, stride: int) -> None:
        iw, ih = self.input_shape

        w_s = (regs[:, 0] + regs[:, 2]) * stride
        h_s = (regs[:, 1] + regs[:, 3]) * stride
        ok_s = float(np.sum((w_s > 2) & (h_s > 2) & (w_s < 2*iw) & (h_s < 2*ih)))

        w_p = regs[:, 0] + regs[:, 2]
        h_p = regs[:, 1] + regs[:, 3]
        ok_p = float(np.sum((w_p > 2) & (h_p > 2) & (w_p < 2*iw) & (h_p < 2*ih)))

        self._ltrb_stride_score += ok_s
        self._ltrb_pixel_score  += ok_p
        self._ltrb_samples_seen += len(regs)

        if self._ltrb_samples_seen >= self._LTRB_MIN_SAMPLES:
            self._ltrb_mode = ('stride' if self._ltrb_stride_score >= self._ltrb_pixel_score
                                else 'pixel')
            print(f"[LTRB auto-detect] LOCKED mode='{self._ltrb_mode}' after "
                  f"{self._ltrb_samples_seen} samples "
                  f"(stride_score={self._ltrb_stride_score:.1f}, "
                  f"pixel_score={self._ltrb_pixel_score:.1f})")

    def predict(self, frame_rgb: np.ndarray) -> List[np.ndarray]:
        """Boxes only. Unchanged public API — kept for existing callers."""
        boxes, _scores = self.predict_with_scores(frame_rgb)
        return boxes

    def predict_with_scores(self, frame_rgb: np.ndarray
                            ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Same inference as predict(), but also returns the per-box detector
        confidence.

        INTEGRATION CHANGE: predict() computed these scores and then threw
        them away, so the only per-target number available downstream was
        the tracker's `quality_score` — a covariance-derived track-health
        measure, NOT a detector confidence. The unified UI must display a
        real "YOLO confidence", and displaying quality_score under that
        label would be a fabricated sensor value. The scores are now
        returned; the decode path itself is untouched.
        """
        # PERF follow-up: _letterbox() writes directly into self._lb_buf,
        # which is a view into self._in_buf[0] — the np.copyto() that used
        # to duplicate the whole padded frame into self._in_buf[0] here is
        # no longer needed (removed; see __init__ for the buffer-aliasing
        # rationale).
        _, ratio, pad_w, pad_h = self._letterbox(frame_rgb)

        # 2. Inference
        raw: dict = self._pipeline.infer(
            {self.input_vstream_info.name: self._in_buf})

        # 3. Parse outputs
        all_boxes:  List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        if self._head_config_valid:
            # --- HEAD_CONFIG path ---
            for bbox_name, cls_name, stride in HEAD_CONFIG:
                bbox_t = raw.get(bbox_name)
                cls_t  = raw.get(cls_name)
                if bbox_t is None or cls_t is None:
                    continue
                bx = bbox_t[0] if bbox_t.ndim == 4 else np.squeeze(bbox_t)
                cx = cls_t[0]  if cls_t.ndim  == 4 else np.squeeze(cls_t)
                bx = self._to_hwc(bx)  # (H, W, 4)
                cx = self._to_hwc(cx)  # (H, W, 1)
                fh, fw, bc = bx.shape
                if bc != 4 or cx.shape[2] != 1:
                    print(f"[WARN] Unexpected channel counts for stride {stride}: "
                          f"bbox_c={bc}, cls_c={cx.shape[2]}. Skipping.")
                    continue

                boxes, scores = self._decode_head(bx, cx, stride, ratio, pad_w, pad_h)
                if boxes is not None:
                    all_boxes.append(boxes)
                    all_scores.append(scores)
        else:
            # --- Auto-detection fallback (shape-based) ---
            heads: dict = {}
            for name, tensor in raw.items():
                arr = tensor[0] if tensor.ndim == 4 else np.squeeze(tensor)
                arr = self._to_hwc(arr)
                if arr.ndim != 3:
                    continue
                fh, fw, ch = arr.shape
                if fh != fw:
                    continue
                stride = self.input_shape[0] // fh
                if stride not in (8, 16, 32):
                    continue
                if stride not in heads:
                    heads[stride] = {}
                if ch == 4:
                    heads[stride]['bbox'] = arr
                elif ch == 1:
                    heads[stride]['cls']  = arr

            for stride, data in heads.items():
                if 'bbox' not in data or 'cls' not in data:
                    continue
                boxes, scores = self._decode_head(
                    data['bbox'], data['cls'], stride, ratio, pad_w, pad_h)
                if boxes is not None:
                    all_boxes.append(boxes)
                    all_scores.append(scores)

        if not all_boxes:
            return [], []

        # 4. NMS
        merged_b = np.vstack(all_boxes)
        merged_s = np.concatenate(all_scores)
        idxs = cv2.dnn.NMSBoxes(
            merged_b.tolist(), merged_s.tolist(),
            self.conf_thresh, self.iou_thresh)
        if len(idxs) == 0:
            return [], []
        keep = idxs.flatten()
        return ([np.array(merged_b[i], dtype=np.float32) for i in keep],
                [float(merged_s[i]) for i in keep])

    def _decode_head(self, bbox_arr: np.ndarray, cls_arr: np.ndarray,
                     stride: int, ratio: float, pad_w: float, pad_h: float
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Decode a single detection head. Returns (boxes, scores) or (None, None)."""
        fh, fw = bbox_arr.shape[0], bbox_arr.shape[1]
        if self._cls_needs_sigmoid is None:
            cmin = float(cls_arr.min()); cmax = float(cls_arr.max())
            self._cls_logit_min = min(self._cls_logit_min, cmin)
            self._cls_logit_max = max(self._cls_logit_max, cmax)
            self._cls_calls_seen += 1
            if self._cls_calls_seen >= self._CLS_MIN_CALLS:
                self._cls_needs_sigmoid = (self._cls_logit_min < 0.0
                                            or self._cls_logit_max > 1.0)
                status = "logits→apply sigmoid" if self._cls_needs_sigmoid \
                         else "already sigmoid in HEF→skip"
                print(f"[Sigmoid auto-detect] LOCKED after "
                      f"{self._cls_calls_seen} calls, range="
                      f"[{self._cls_logit_min:.3f}, {self._cls_logit_max:.3f}]"
                      f" → {status}")

        cls_needs_sigmoid = self._cls_needs_sigmoid \
            if self._cls_needs_sigmoid is not None else False
        if cls_needs_sigmoid:
            cls_2d = 1.0 / (1.0 + np.exp(-np.clip(cls_arr[:, :, 0], -88.0, 88.0)))
        else:
            cls_2d = cls_arr[:, :, 0]

        mask = cls_2d > self.conf_thresh
        if not np.any(mask):
            return None, None

        gy, gx   = np.where(mask)
        scores   = cls_2d[mask].astype(np.float32)
        regs     = bbox_arr[gy, gx]                   # (N, 4) LTRB

        cx_grid  = (gx.astype(np.float32) + 0.5) * stride
        cy_grid  = (gy.astype(np.float32) + 0.5) * stride
        if self._ltrb_mode is None and len(gx) > 0:
            self._update_ltrb_vote(regs, stride)

        # Default while still accumulating votes: 'stride' — the standard
        # Ultralytics anchor-free LTRB-in-stride-units convention (also
        # this file's own original tie-break preference when scores were
        # equal), consistent with the "one2one_cv2/cv3" head naming
        # verified against this exact HEF in TWO_CAMERAS_AUDIT.md Section 9.
        ltrb_mode = self._ltrb_mode if self._ltrb_mode is not None else 'stride'

        if ltrb_mode == 'pixel':
            # LTRB already in pixel coordinates
            x1p = cx_grid - regs[:, 0]
            y1p = cy_grid - regs[:, 1]
            x2p = cx_grid + regs[:, 2]
            y2p = cy_grid + regs[:, 3]
        else:
            x1p = cx_grid - regs[:, 0] * stride
            y1p = cy_grid - regs[:, 1] * stride
            x2p = cx_grid + regs[:, 2] * stride
            y2p = cy_grid + regs[:, 3] * stride

        x1o = (x1p - pad_w) / ratio
        y1o = (y1p - pad_h) / ratio
        x2o = (x2p - pad_w) / ratio
        y2o = (y2p - pad_h) / ratio

        wo = x2o - x1o
        ho = y2o - y1o
        return np.column_stack((x1o, y1o, wo, ho)).astype(np.float32), scores

    def release(self):
        try:
            self._vstreams.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._act_ctx.__exit__(None, None, None)
        except Exception:
            pass
def _check_camera_manager_is_current():

    params = inspect.signature(CameraManager.__init__).parameters
    if "max_fps" not in params:
        raise SystemExit(
            "\n[SETUP ERROR] The camera_manager.py being imported is OUT OF DATE.\n"
            f"  Loaded from : {inspect.getfile(CameraManager)}\n"
            "  Problem     : it does not accept the 'max_fps' argument that\n"
            "                this script passes to CameraManager(...).\n"
            "  Fix         : copy the updated camera_manager.py into that same\n"
            "                folder, then run again.\n"
            "  Note        : this script and camera_manager.py must ALWAYS be\n"
            "                updated together.\n"
        )


def main():
    _check_camera_manager_is_current()

    camera_manager = CameraManager(
        width=640,
        height=480,
        fps=30,
        # REVERTED from 60 back to None (= hard 30 fps lock).
        # Allowing the sensor to run faster shortens the maximum exposure
        # time, and the auto-exposure compensates with higher gain, i.e. a
        # noisier image. Detection quality depends directly on that image
        # — it is the ONLY change in this file that touches what the
        # sensor actually produces (the tracker and ego-motion changes
        # cannot affect detection, since detector.predict() runs on the
        # raw camera frame). Detection reliability outweighs display FPS.
        max_fps=None,
        buffer_count=4,
        debounce_interval=0.5,
        warmup_frames=3,
    )

    detector = HailoInference(
        "bestty_yolo260508.hef",
        conf_thresh=0.4,
        iou_thresh=0.45,
    )

    tracker = AdvancedADASTracker()

    yolo_fps_ema = 0.0
    tracker_fps_ema = 0.0
    display_fps_ema = 0.0

    alpha_fps = 0.9
    frame_count = 0
    frame_skip = 2

    # Held by the 'f' key — see the switch block below. Off by default,
    # so normal operation is unaffected.
    switch_frozen = False

    try:
        while True:
            frame_start = time.time()
            frame_count += 1

            frame_rgb = camera_manager.get_frame()

            if frame_rgb is None:
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            detections = []
            has_confirmed_track = any(
                t.state == TrackState.Confirmed for t in tracker.tracks)
            run_detector = (frame_count % frame_skip == 0) or not has_confirmed_track

            t0 = time.time()

            if run_detector:
                detections = detector.predict(frame_rgb)

            yolo_time = time.time() - t0

            t1 = time.time()

            tracker.process_frame(
                frame_bgr,
                detections,
                run_association=run_detector
            )

            tracker_time = time.time() - t1

            # Read the active camera BEFORE the draw loop: the per-track
            # distance estimate needs to know which camera's optics
            # produced this frame, because focal length is per-camera.
            # The switch decision further below still reads the same
            # value — nothing between here and there changes it.
            current_camera = camera_manager.get_active_camera()

            max_height = 0.0
            # Width of the LARGEST live box = the closest target. Drives
            # both the distance-based camera switch and the 'c'
            # calibration key.
            max_width_px = 0.0

            for track in tracker.tracks:

                if track.state == TrackState.Deleted:
                    continue

                # PERF follow-up: cached value — see ADAsTrack.__init__.
                # Safe here because nothing mutates any track's IMM state
                # between process_frame() returning and this loop running.
                state, _ = track.merged_x, track.merged_P

                if state is None:
                    continue

                if len(state) < 4:
                    continue

                if not np.all(np.isfinite(state[:4])):
                    continue

                cx, cy, w, h = map(float, state[:4])

                if w <= 1 or h <= 1:
                    continue

                if track.state in (
                    TrackState.Confirmed,
                    TrackState.Tentative,
                ):
                    max_height = max(max_height, h)
                    max_width_px = max(max_width_px, w)

                x1 = int(round(cx - w / 2))
                y1 = int(round(cy - h / 2))
                x2 = int(round(cx + w / 2))
                y2 = int(round(cy + h / 2))

                x1 = max(0, min(frame_bgr.shape[1] - 1, x1))
                y1 = max(0, min(frame_bgr.shape[0] - 1, y1))
                x2 = max(0, min(frame_bgr.shape[1] - 1, x2))
                y2 = max(0, min(frame_bgr.shape[0] - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                if track.state == TrackState.Confirmed:
                    color = (0, 255, 0)
                elif track.state == TrackState.Tentative:
                    color = (0, 200, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame_bgr,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                score = getattr(track, "quality_score", 1.0)

                # Distance label. estimate_distance_m() returns None when
                # the pinhole constants have not been measured, in which
                # case we fall back to the original quality score rather
                # than printing a made-up number.
                dist_m = estimate_distance_m(w, current_camera)
                label = (f"Drone {dist_m:.1f}m" if dist_m is not None
                         else f"Drone {score:.2f}")

                cv2.putText(
                    frame_bgr,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            target_distance_m = estimate_distance_m(max_width_px, current_camera)

            if switch_frozen:
                # Switching held by the 'f' key. Needed for calibration:
                # CALIBRATION_DISTANCE_M and SWITCH_TO_NEAR_BELOW_M are
                # both 1.5 m, so without this the camera would switch out
                # from under you at exactly the moment you press 'c'.
                pass

            elif target_distance_m is not None:
                # PRIMARY PATH — distance-based switching.
                if (
                    current_camera == CameraManager.FAR_CAMERA_ID
                    and target_distance_m < SWITCH_TO_NEAR_BELOW_M
                ):
                    # AUDIT BUG #2: use the return value, so a switch
                    # suppressed by the debounce is distinguishable from
                    # one that actually happened.
                    if camera_manager.switch_to_near():
                        print(f"[CameraSwitch] FAR -> NEAR "
                              f"({target_distance_m:.1f} m)")

                elif (
                    current_camera == CameraManager.NEAR_CAMERA_ID
                    and target_distance_m > SWITCH_TO_FAR_ABOVE_M
                ):
                    if camera_manager.switch_to_far():
                        print(f"[CameraSwitch] NEAR -> FAR "
                              f"({target_distance_m:.1f} m)")

            elif max_height > 0:
                if (
                    current_camera == CameraManager.FAR_CAMERA_ID
                    and max_height > THRESHOLD_NEAR_HEIGHT
                ):
                    if camera_manager.switch_to_near():
                        print("[CameraSwitch] FAR -> NEAR (px fallback)")

                elif (
                    current_camera == CameraManager.NEAR_CAMERA_ID
                    and max_height < THRESHOLD_FAR_HEIGHT
                ):
                    if camera_manager.switch_to_far():
                        print("[CameraSwitch] NEAR -> FAR (px fallback)")

            if run_detector and yolo_time > 0:
                yolo_fps = 1.0 / yolo_time
                yolo_fps_ema = (
                    alpha_fps * yolo_fps_ema
                    + (1.0 - alpha_fps) * yolo_fps
                )

            tracker_fps = 1.0 / max(tracker_time, 1e-6)
            tracker_fps_ema = (
                alpha_fps * tracker_fps_ema
                + (1.0 - alpha_fps) * tracker_fps
            )

            display_time = time.time() - frame_start
            display_fps = 1.0 / max(display_time, 1e-6)

            display_fps_ema = (
                alpha_fps * display_fps_ema
                + (1.0 - alpha_fps) * display_fps
            )

            if current_camera == CameraManager.FAR_CAMERA_ID:
                cam_name = "IMX477 FAR"
            else:
                cam_name = "IMX708 NEAR"

            cv2.putText(
                frame_bgr,
                f"YOLO    : {yolo_fps_ema:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame_bgr,
                f"Tracker : {tracker_fps_ema:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame_bgr,
                f"Display : {display_fps_ema:.1f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame_bgr,
                f"Camera  : {cam_name}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Make the uncalibrated state explicit rather than silently
            # omitting distances, so a missing constant is never mistaken
            # for "the drone is out of range".
            if DRONE_REAL_WIDTH_M is None or not CAMERA_FOCAL_PX.get(current_camera):
                cv2.putText(
                    frame_bgr,
                    "Distance: NOT CALIBRATED (press c)",
                    (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            if switch_frozen:
                cv2.putText(
                    frame_bgr,
                    "SWITCHING FROZEN (press f)",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2,
                )

            cv2.imshow("ADAS", frame_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("f"):
                switch_frozen = not switch_frozen
                print(f"[CameraSwitch] switching "
                      f"{'FROZEN — camera will not change' if switch_frozen else 'RESUMED'}")

            if key == ord("c"):
                if DRONE_REAL_WIDTH_M is None:
                    print("[Calibration] DRONE_REAL_WIDTH_M is not set — "
                          "measure the drone's real width (m) and set it "
                          "at the top of this file first.")
                elif max_width_px <= 1.0:
                    print("[Calibration] no drone box visible right now — "
                          "get a stable detection first, then press 'c'.")
                else:
                    focal_px = (max_width_px * CALIBRATION_DISTANCE_M
                                / DRONE_REAL_WIDTH_M)
                    print(f"[Calibration] camera={current_camera} "
                          f"({cam_name}): box width {max_width_px:.1f}px "
                          f"at {CALIBRATION_DISTANCE_M} m "
                          f"with real width {DRONE_REAL_WIDTH_M} m")
                    print(f"[Calibration]   -> set "
                          f"CAMERA_FOCAL_PX[{current_camera}] = "
                          f"{focal_px:.1f}")
                    print("[Calibration] Repeat this on the OTHER camera "
                          "too — the two have different optics, so they "
                          "need separate focal lengths.")

    finally:
        detector.release()
        camera_manager.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()