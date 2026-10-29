# Trajectory answer formats: planning trajectory (6 waypoints @0.5 s or 12 @0.25 s) and optional driving path.
# Numpy only: also loaded by file path from the eval scripts (no plugin import).
import re

import numpy as np

PATH_RES = 0.5  # arc-length resolution (m) of the GT path stored in the traj_extra sidecar
DENSE_KEY = 'dense planning trajectory'
PATH_KEY = 'driving path'
_TRAJ_RE = re.compile(r'trajectory\s*\[([^\]]*)')
_PATH_RE = re.compile(PATH_KEY + r'\s*\[([^\]]*)')
_COORD_RE = re.compile(r'\(\s*\+?(-?\d*\.?\d+)\s*,\s*\+?(-?\d*\.?\d+)\s*\)')


def num_plan_coords(fmt):
    """Number of answer coordinates of the planning turn for a traj_format dict (None -> 6)."""
    if not fmt:
        return 6
    return fmt['num_wp'] + (fmt['path_points'] if fmt.get('path_spacing', 0) > 0 else 0)


def half_second_index(num_wp):
    """Indices of the 0.5 s waypoints (the evaluated ones) within the trajectory."""
    return list(range(6)) if num_wp == 6 else list(range(1, 12, 2))


def path_index(spacing, points):
    """Indices into the stored PATH_RES path for `points` points every `spacing` m."""
    return [int(round(k * spacing / PATH_RES)) - 1 for k in range(1, points + 1)]


def is_new_format(text):
    return isinstance(text, str) and (DENSE_KEY in text or PATH_KEY in text)


def _coords(s):
    return np.array([[float(x), float(y)] for x, y in _COORD_RE.findall(s)], dtype=np.float64).reshape(-1, 2)


def split_answer(text):
    """-> (trajectory (K, 2) or None, path (N, 2) or None, dense)."""
    m = _TRAJ_RE.search(text)
    p = _PATH_RE.search(text)
    traj = _coords(m.group(1)) if m else None
    path = _coords(p.group(1)) if p else None
    return traj, path, DENSE_KEY in text


def to_half_seconds(traj, dense):
    """The six 0.5 s waypoints of a parsed trajectory, or None if it is too short."""
    if traj is None:
        return None
    n = 12 if dense else 6
    return traj[half_second_index(n)] if len(traj) >= n else None


def resample_polyline(points, s, extrap_dir=None):
    """Points at arc lengths `s` along the polyline (0, 0) -> points; beyond its end, continue along
    `extrap_dir` (default: direction of the last non-degenerate segment, else +x)."""
    poly = np.vstack([np.zeros((1, 2)), np.asarray(points, dtype=np.float64).reshape(-1, 2)])
    seg = np.diff(poly, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-6
    starts, seg, seg_len = poly[:-1][keep], seg[keep], seg_len[keep]
    s = np.asarray(s, dtype=np.float64)
    if extrap_dir is None:
        extrap_dir = seg[-1] / seg_len[-1] if len(seg) else np.array([1.0, 0.0])
    extrap_dir = np.asarray(extrap_dir, dtype=np.float64)
    extrap_dir = extrap_dir / max(np.linalg.norm(extrap_dir), 1e-9)
    if not len(seg):
        return s[:, None] * extrap_dir[None]
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    i = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(seg) - 1)
    inside = starts[i] + ((s - cum[i]) / seg_len[i])[:, None] * seg[i]
    end = starts[-1] + seg[-1]
    beyond = end[None] + (s - cum[-1])[:, None] * extrap_dir[None]
    return np.where((s <= cum[-1])[:, None], inside, beyond)


def project_on_path(traj, path, keep_idx=None):
    """Keep the distance travelled along `traj` (from the origin) at the waypoints `keep_idx`, but place each
    point on the path. Falls back to the trajectory if the path is empty."""
    traj = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
    keep_idx = list(range(len(traj))) if keep_idx is None else keep_idx
    if path is None or len(path) == 0:
        return traj[keep_idx]
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), traj]), axis=0), axis=1)
    return resample_polyline(path, np.cumsum(steps)[keep_idx])
