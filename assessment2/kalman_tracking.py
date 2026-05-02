#Object Tracking: Kalman Filter for Translation and Rotation
#Tools used in this file:
#Tool: GitHub CoPilot, for code completion
#for example function body, variable and function names

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE   = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(BASE, 'parachute', 'parachute', 'GT')
OUT_DIR = os.path.join(BASE, 'output_tracking')
os.makedirs(OUT_DIR, exist_ok=True)

N_FRAMES       = 51
TRAIN_FRAMES   = 41    # frames 0–40 used for filter initialisation/tuning
PREDICT_FRAMES = 10    # frames 41–50 evaluated
PARACHUTE_LABEL = 255



#measurement extraction from GT masks


def load_mask(frame_idx: int) -> np.ndarray:
    path = os.path.join(GT_DIR, f'parachute_{frame_idx:05d}.png')
    gt   = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(path)
    return (gt == PARACHUTE_LABEL).astype(np.uint8) * 255


def compute_centroid(mask: np.ndarray):
    """Return (cx, cy) centroid of the foreground region."""
    M = cv2.moments(mask)
    if M['m00'] == 0:
        return np.nan, np.nan
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    return cx, cy


def compute_orientation(mask: np.ndarray) -> float:
    """
    Estimate the parachute's orientation angle θ (degrees) using second-order
    central moments.

    The covariance matrix of the region is:
        Σ = [[μ20, μ11], [μ11, μ02]] / m00

    The principal axis direction (angle of the major axis) is:
        θ = 0.5 · atan2(2·μ11, μ20 - μ02)

    This gives θ in [-90°, 90°], i.e. unsigned/modulo-180°.
    Returns degrees in [0°, 180°) for consistency.
    """
    M = cv2.moments(mask)
    if M['m00'] == 0:
        return np.nan
    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    theta_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    theta_deg = np.degrees(theta_rad)   # in [-90°, 90°]
    # Map to [0°, 180°)
    if theta_deg < 0:
        theta_deg += 180.0
    return theta_deg


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap a sequence of angles (in degrees, [0°, 180°) range) to produce a
    smooth continuous signal by resolving the 180° ambiguity.

    At each step we choose the continuation that minimises |Δθ|.
    """
    unwrapped = angles.copy().astype(float)
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i-1]
        #resolve 180° ambiguity: prefer the smallest signed step
        while diff >  90.0:
            diff -= 180.0
        while diff < -90.0:
            diff += 180.0
        unwrapped[i] = unwrapped[i-1] + diff
    return unwrapped


def extract_all_measurements() -> tuple:
    """
    Returns:
        centroids  - (N, 2) array of (cx, cy)
        angles_raw - (N,)   array of θ in [0°, 180°)
        angles_uw  - (N,)   unwrapped angle sequence
    """
    centroids  = np.zeros((N_FRAMES, 2))
    angles_raw = np.zeros(N_FRAMES)
    for i in range(N_FRAMES):
        mask         = load_mask(i)
        cx, cy       = compute_centroid(mask)
        centroids[i] = [cx, cy]
        angles_raw[i]= compute_orientation(mask)
    angles_uw = unwrap_angles(angles_raw)
    return centroids, angles_raw, angles_uw


#kalman filter - generic constant-velocity implementation

class KalmanFilter1D:
    """
    Scalar constant-velocity Kalman filter.

    State vector:   x = [position, velocity]ᵀ
    Observation:    z = position  (scalar)

    Matrices
    --------
    F (transition):  [[1, dt], [0, 1]]
    H (observation): [[1, 0]]
    Q (process noise):  diagonal, tuned from training residuals
    R (measurement noise): scalar, tuned from training residuals
    """

    def __init__(self, dt: float = 1.0, q_pos: float = 1.0,
                 q_vel: float = 0.5, r: float = 5.0):
        self.dt = dt
        self.F  = np.array([[1.0, dt],
                             [0.0, 1.0]])
        self.H  = np.array([[1.0, 0.0]])
        self.Q  = np.diag([q_pos, q_vel])
        self.R  = np.array([[r]])
        #state and covariance - initialised on first update
        self.x  = None   # (2,)
        self.P  = None   # (2,2)

    def initialise(self, pos0: float, vel0: float = 0.0,
                   p_pos: float = 10.0, p_vel: float = 1.0) -> None:
        self.x = np.array([pos0, vel0])
        self.P = np.diag([p_pos, p_vel])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0])

    def update(self, z: float) -> float:
        """Correct with measurement z; return updated position estimate."""
        y = float(z) - float((self.H @ self.x).item())   # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0])


class KalmanFilter2D:
    """
    Two-channel constant-velocity Kalman filter for (cx, cy).

    Treats x and y as independent 1-D filters sharing the same noise params.
    """

    def __init__(self, dt=1.0, q_pos=1.0, q_vel=0.5, r=5.0):
        self.kf_x = KalmanFilter1D(dt, q_pos, q_vel, r)
        self.kf_y = KalmanFilter1D(dt, q_pos, q_vel, r)

    def initialise(self, cx, cy, vx=0.0, vy=0.0,
                   p_pos=10.0, p_vel=1.0):
        self.kf_x.initialise(cx, vx, p_pos, p_vel)
        self.kf_y.initialise(cy, vy, p_pos, p_vel)

    def predict(self):
        return self.kf_x.predict(), self.kf_y.predict()

    def update(self, cx, cy):
        return self.kf_x.update(cx), self.kf_y.update(cy)


#noise parameter tuning (from training residuals)
def estimate_noise_params(signal: np.ndarray, dt: float = 1.0) -> dict:
    """
    Estimate R and Q from training data.

    R ≈ variance of first-difference residuals (measurement noise proxy)
    Q_pos, Q_vel estimated via Allan variance / innovation variance heuristic.
    """
    diffs    = np.diff(signal)
    r        = float(np.var(diffs) / 2.0)      # measurement noise
    accel    = np.diff(diffs)                   # second differences ≈ acceleration
    q_vel    = float(np.var(accel))
    q_pos    = q_vel * (dt ** 2) / 4.0
    return dict(r=max(r, 0.01), q_pos=max(q_pos, 0.01), q_vel=max(q_vel, 0.01))


#run translation kalman filter
def run_translation_filter(centroids: np.ndarray):
    """
    Train on frames 0-40, predict frames 41-50


    Returns:
        filtered   - (41, 2) filtered positions during training phase
        predicted  - (10, 2) predicted positions for frames 41-50
    """
    train = centroids[:TRAIN_FRAMES]

    #tune noise from training data
    px = estimate_noise_params(train[:, 0])
    py = estimate_noise_params(train[:, 1])
    q_pos = (px['q_pos'] + py['q_pos']) / 2
    q_vel = (px['q_vel'] + py['q_vel']) / 2
    r     = (px['r']     + py['r'])     / 2

    kf = KalmanFilter2D(dt=1.0, q_pos=q_pos, q_vel=q_vel, r=r)

    #initialise with frame 0, estimate initial velocity from frames 0-2
    vx0 = float(np.mean(np.diff(train[:3, 0])))
    vy0 = float(np.mean(np.diff(train[:3, 1])))
    kf.initialise(train[0, 0], train[0, 1], vx0, vy0, p_pos=100.0, p_vel=10.0)

    #filter pass over training frames
    filtered = np.zeros((TRAIN_FRAMES, 2))
    for i, (cx, cy) in enumerate(train):
        kf.predict()
        filtered[i] = kf.update(cx, cy)

    #pure prediction for frames 41–50 (no measurement updates)
    predicted = np.zeros((PREDICT_FRAMES, 2))
    for i in range(PREDICT_FRAMES):
        px_pred, py_pred = kf.predict()
        predicted[i] = [px_pred, py_pred]

    return filtered, predicted


#run rotation kalman filter


def run_rotation_filter(angles_uw: np.ndarray):
    """
    Train on frames 0-40 using unwrapped angles, predict frames 41-50.

    Returns:
        filtered   - (41,) filtered angle estimates
        predicted  - (10,) predicted angles
    """
    train = angles_uw[:TRAIN_FRAMES]

    params = estimate_noise_params(train)
    kf = KalmanFilter1D(dt=1.0,
                        q_pos=params['q_pos'],
                        q_vel=params['q_vel'],
                        r=params['r'])

    omega0 = float(np.mean(np.diff(train[:3])))
    kf.initialise(pos0=train[0], vel0=omega0, p_pos=100.0, p_vel=10.0)

    filtered = np.zeros(TRAIN_FRAMES)
    for i, theta in enumerate(train):
        kf.predict()
        filtered[i] = kf.update(theta)

    predicted = np.zeros(PREDICT_FRAMES)
    for i in range(PREDICT_FRAMES):
        predicted[i] = kf.predict()

    return filtered, predicted



#error computation
def angular_error(kf_angle: float, gt_angle: float) -> float:
    """
    Smallest angular difference in [0°, 90°] accounting for 180° ambiguity.
    e_θ = min(|Δθ|, 180 - |Δθ|)
    """
    delta = abs(kf_angle - gt_angle) % 180.0
    return min(delta, 180.0 - delta)


def compute_errors(centroids, angles_raw,
                   pos_predicted, angle_predicted_uw):
    """
    Returns arrays of translation and rotation errors for frames 41-50.
    """
    gt_centroids = centroids[TRAIN_FRAMES:]          # (10, 2)
    gt_angles    = angles_raw[TRAIN_FRAMES:]         # (10,)  raw [0°, 180°)

    #wrap predicted (unwrapped) angles back to (0°, 180°) for error computation
    pred_angles_wrapped = angle_predicted_uw % 180.0

    e_pos = np.sqrt(np.sum((pos_predicted - gt_centroids) ** 2, axis=1))
    e_ang = np.array([angular_error(float(p), float(g))
                      for p, g in zip(pred_angles_wrapped, gt_angles)])

    return e_pos, e_ang, gt_centroids, gt_angles

#plots and tables
def print_error_table(e_pos, e_ang):
    print('\n=== Kalman Filter Prediction Errors (Frames 41-50) ===')
    print(f'{"Frame":>6}  {"Translation Error (px)":>22}  {"Rotation Error (°)":>18}')
    print('-' * 54)
    for i, (ep, ea) in enumerate(zip(e_pos, e_ang)):
        print(f'{TRAIN_FRAMES + i:6d}  {ep:22.3f}  {ea:18.3f}')
    print('-' * 54)
    print(f'{"Mean":>6}  {e_pos.mean():22.3f}  {e_ang.mean():18.3f}')
    print(f'{"Std":>6}  {e_pos.std():22.3f}  {e_ang.std():18.3f}')


def plot_translation_tracking(centroids, pos_filtered, pos_predicted):
    frames_train   = np.arange(TRAIN_FRAMES)
    frames_predict = np.arange(TRAIN_FRAMES, N_FRAMES)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Translation Kalman Filter — Centroid Tracking',
                 fontsize=12, fontweight='bold')

    for ax, dim, label in zip(axes, [0, 1], ['cx (horizontal)', 'cy (vertical)']):
        ax.plot(np.arange(N_FRAMES), centroids[:, dim],
                'k.', markersize=4, label='GT measurement')
        ax.plot(frames_train, pos_filtered[:, dim],
                'steelblue', linewidth=1.5, label='KF filtered (train)')
        ax.plot(frames_predict, pos_predicted[:, dim],
                'firebrick', linewidth=1.5, linestyle='--', label='KF predicted (test)')
        ax.axvline(x=TRAIN_FRAMES - 0.5, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Frame index', fontsize=10)
        ax.set_ylabel(f'{label} (pixels)', fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'translation_tracking.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_rotation_tracking(angles_raw, angles_uw, rot_filtered, rot_predicted_uw):
    frames_train   = np.arange(TRAIN_FRAMES)
    frames_predict = np.arange(TRAIN_FRAMES, N_FRAMES)
    rot_predicted_wrapped = rot_predicted_uw % 180.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Rotation Kalman Filter — Orientation Tracking',
                 fontsize=12, fontweight='bold')

    # Unwrapped view
    ax = axes[0]
    ax.plot(np.arange(N_FRAMES), angles_uw, 'k.', markersize=4, label='GT (unwrapped)')
    ax.plot(frames_train, rot_filtered,
            'steelblue', linewidth=1.5, label='KF filtered (train)')
    ax.plot(frames_predict, rot_predicted_uw,
            'firebrick', linewidth=1.5, linestyle='--', label='KF predicted (test)')
    ax.axvline(x=TRAIN_FRAMES - 0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Frame index', fontsize=10)
    ax.set_ylabel('Orientation θ (°, unwrapped)', fontsize=10)
    ax.set_title('Unwrapped angle', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Raw/wrapped view
    ax = axes[1]
    ax.plot(np.arange(N_FRAMES), angles_raw, 'k.', markersize=4, label='GT raw [0°,180°)')
    ax.plot(frames_predict, rot_predicted_wrapped,
            'firebrick', linewidth=1.5, linestyle='--', label='KF predicted (test)')
    ax.axvline(x=TRAIN_FRAMES - 0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Frame index', fontsize=10)
    ax.set_ylabel('Orientation θ (°)', fontsize=10)
    ax.set_title('Wrapped angle [0°, 180°)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'rotation_tracking.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_errors(e_pos, e_ang):
    frames = np.arange(TRAIN_FRAMES, N_FRAMES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Kalman Filter Prediction Errors — Frames 41-50',
                 fontsize=12, fontweight='bold')

    axes[0].bar(frames, e_pos, color='steelblue', edgecolor='navy', linewidth=0.5)
    axes[0].axhline(e_pos.mean(), color='firebrick', linestyle='--',
                    linewidth=1.2, label=f'Mean = {e_pos.mean():.2f} px')
    axes[0].set_xlabel('Frame index', fontsize=10)
    axes[0].set_ylabel('Translation error (pixels)', fontsize=10)
    axes[0].set_title('Translation Error epos', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(frames, e_ang, color='seagreen', edgecolor='darkgreen', linewidth=0.5)
    axes[1].axhline(e_ang.mean(), color='firebrick', linestyle='--',
                    linewidth=1.2, label=f'Mean = {e_ang.mean():.2f}°')
    axes[1].set_xlabel('Frame index', fontsize=10)
    axes[1].set_ylabel('Rotation error (°)', fontsize=10)
    axes[1].set_title('Rotation Error eθ', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'prediction_errors.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_trajectory(centroids, pos_filtered, pos_predicted):
    """2-D trajectory plot in image coordinates."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(centroids[:TRAIN_FRAMES, 0], centroids[:TRAIN_FRAMES, 1],
               c='steelblue', s=20, label='GT train centroids', zorder=3)
    ax.scatter(centroids[TRAIN_FRAMES:, 0], centroids[TRAIN_FRAMES:, 1],
               c='black', s=25, marker='x', linewidths=1.2,
               label='GT test centroids', zorder=3)
    ax.plot(pos_filtered[:, 0], pos_filtered[:, 1],
            'steelblue', linewidth=1.2, alpha=0.6, label='KF filtered (train)')
    ax.plot(pos_predicted[:, 0], pos_predicted[:, 1],
            'firebrick', linewidth=1.8, linestyle='--', label='KF predicted (test)')

    # Annotate first and last predicted point
    ax.annotate('f=41', pos_predicted[0], fontsize=8, color='firebrick')
    ax.annotate('f=50', pos_predicted[-1], fontsize=8, color='firebrick')

    ax.invert_yaxis()
    ax.set_xlabel('x (pixels)', fontsize=10)
    ax.set_ylabel('y (pixels)', fontsize=10)
    ax.set_title('Parachute 2-D Trajectory', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'trajectory.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def save_error_csv(e_pos, e_ang):
    rows = []
    for i, (ep, ea) in enumerate(zip(e_pos, e_ang)):
        rows.append(f'{TRAIN_FRAMES + i},{ep:.4f},{ea:.4f}')
    csv_path = os.path.join(OUT_DIR, 'prediction_errors.csv')
    with open(csv_path, 'w') as f:
        f.write('frame,translation_error_px,rotation_error_deg\n')
        f.write('\n'.join(rows))
    print(f'Saved: {csv_path}')

#main
if __name__ == '__main__':
    print('--- Object Tracking with Kalman Filters -------------------------')

    print('\n[1/5] Extracting measurements from GT masks ...')
    centroids, angles_raw, angles_uw = extract_all_measurements()
    print(f'      Centroid range: cx=[{centroids[:,0].min():.1f}, {centroids[:,0].max():.1f}]  '
          f'cy=[{centroids[:,1].min():.1f}, {centroids[:,1].max():.1f}]')
    print(f'      Angle range (raw): [{angles_raw.min():.1f}°, {angles_raw.max():.1f}°]')
    print(f'      Angle range (unwrapped): [{angles_uw.min():.1f}°, {angles_uw.max():.1f}°]')

    print('\n[2/5] Running translation Kalman filter ...')
    pos_filtered, pos_predicted = run_translation_filter(centroids)

    print('\n[3/5] Running rotation Kalman filter ...')
    rot_filtered, rot_predicted_uw = run_rotation_filter(angles_uw)

    print('\n[4/5] Computing errors and producing plots ...')
    e_pos, e_ang, gt_centroids, gt_angles = compute_errors(
        centroids, angles_raw, pos_predicted, rot_predicted_uw)

    print_error_table(e_pos, e_ang)
    save_error_csv(e_pos, e_ang)

    plot_translation_tracking(centroids, pos_filtered, pos_predicted)
    plot_rotation_tracking(angles_raw, angles_uw, rot_filtered, rot_predicted_uw)
    plot_errors(e_pos, e_ang)
    plot_trajectory(centroids, pos_filtered, pos_predicted)

    print('\n[5/5] Object tracking complete.')
    print(f'Output files are in: {OUT_DIR}')