import random
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from DifferentialDrive import DifferentialDrive
from UKF_SLAM_CLASSES import UKF_SLAM, UKF_SLAM_DA
from utils.angle import normalize_angle

SEED = 16

# ==========================================
# SIMULATION CASES
# ==========================================
# --- Best Case 
#NUM_LANDMARKS = 35
#noise = 0.05


# --- Worst Case
NUM_LANDMARKS = 4
noise = 0.2

# ==========================================
# EXECUTION MODES
# ==========================================
MODE_PERFECT_ID_ONLY = False
MODE_DATA_ASSOCIATION_ONLY = False
MODE_COMPARISON = True

# ==========================================
# PLOT OPTIONS
# ==========================================
ENABLE_REALTIME = True
ENABLE_RESULTS = True
ENABLE_ERRORS = True
ENABLE_LANDMARKERROR = True
ENABLE_LANDMARK_ERROR_HISTORY = True 

# ==========================================
# SIMULATION SETTINGS
# ==========================================
CIRCLE_TRAJECTORY = False
INFTY_TRAJECTORY = True
ENABLED_FOV = False

RANGE_MIN = -10.0
RANGE_MAX = 10.0
ROBOT_RANGE = 5

# ==========================================
# NOISE SETTINGS
# ==========================================
Q = 0.01 * np.eye(2)
R = np.diag([noise, noise, noise / 10])

CHI2_95 = 5.991


def plot_covariance_ellipse(x, y, cov, ax, color, alpha=0.3):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(CHI2_95 * eigvals)
    ellipse = Ellipse(xy=(x, y), width=width, height=height, angle=angle, edgecolor=color, fc=color, alpha=alpha, lw=1.5)
    ax.add_patch(ellipse)

def get_command(t, steps):
    u_true = {}
    if INFTY_TRAJECTORY:
        v = 0.4
        w = 0.126
        if t < steps // 2:
            u_true = {'r1': w, 't': v, 'r2': 0.0}
        else:
            u_true = {'r1': -w, 't': v, 'r2': 0.0}
    if CIRCLE_TRAJECTORY:
        v = 0.4
        w = 0.0628
        u_true = {'r1': w, 't': v, 'r2': 0.0}
    return u_true

def get_initial_conditions(P):
    x0 = None
    if INFTY_TRAJECTORY:
        x0 = np.array([0.0, 0.0, 0.0])
    if CIRCLE_TRAJECTORY:
        x0 = np.array([0.0, -7.0, 0.0])
    x_hat = scipy.linalg.sqrtm(P) @ np.random.randn(x0.shape[0], 1) + x0.reshape(-1, 1)
    return x0, x_hat

def get_fov(enabled):
    return np.pi / 2 if enabled else np.pi

def get_steps():
    if INFTY_TRAJECTORY: 
        return 100
    if CIRCLE_TRAJECTORY: 
        return 200
    return None

def generate_landmarks(n):
    landmarks = {}
    for i in range(1, n + 1):
        x = round(random.uniform(RANGE_MIN, RANGE_MAX), 1)
        y = round(random.uniform(RANGE_MIN, RANGE_MAX), 1)
        landmarks[i] = (x, y)
    return landmarks

def draw_realtime_ax(ax, enable_da, true_landmarks, true_history, path_history, x_true, x_ukf, P, map_list):
    ax.clear()
    ax.set_title(f"UKF-SLAM {'with Data Association' if enable_da else 'with Perfect IDs'}", fontsize=14)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(RANGE_MIN - 1, RANGE_MAX + 1)
    ax.set_ylim(RANGE_MIN - 1, RANGE_MAX + 1)

    for l_id, (lx, ly) in true_landmarks.items():
        ax.plot(lx, ly, 'k*', markersize=12, label="True Landmark" if l_id == 1 else "")
        ax.text(lx + 0.15, ly + 0.15, str(l_id), color='black', fontsize=9, fontweight='bold')

    if len(true_history) > 1:
        px, py = zip(*true_history)
        ax.plot(px, py, 'g--', alpha=0.6, label="True Path")

    if len(path_history) > 1:
        px, py = zip(*path_history)
        ax.plot(px, py, 'r--', alpha=0.6, label="Estimated Path")

    ax.plot(x_true[0], x_true[1], 'go', markersize=8, label="True Robot")
    ax.plot(x_ukf[0, 0], x_ukf[1, 0], 'ro', markersize=8, label="Estimated Robot")
    ax.quiver(x_ukf[0, 0], x_ukf[1, 0], np.cos(x_ukf[2, 0]), np.sin(x_ukf[2, 0]), color='r', scale=15)
    ax.quiver(x_true[0], x_true[1], np.cos(x_true[2]), np.sin(x_true[2]), color='g', scale=15)

    plot_covariance_ellipse(x_ukf[0, 0], x_ukf[1, 0], P[0:2, 0:2], ax, color='red')

    for i, l_id in enumerate(map_list):
        lx, ly = x_ukf[3 + 2 * i, 0], x_ukf[3 + 2 * i + 1, 0]
        if enable_da and l_id < 0:
            color, label = 'orange', "Est. Landmark (new)" if i == 0 else ""
        else:
            color, label = 'blue', "Est. Landmark" if i == 0 else ""
        ax.plot(lx, ly, 'o', color=color, markersize=6, label=label)
        lm_cov = P[3 + 2 * i: 3 + 2 * i + 2, 3 + 2 * i: 3 + 2 * i + 2]
        plot_covariance_ellipse(lx, ly, lm_cov, ax, color=color, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    fov_angle = get_fov(ENABLED_FOV)
    ang_l, ang_r = x_true[2] + fov_angle, x_true[2] - fov_angle
    ax.plot([x_true[0], x_true[0] + ROBOT_RANGE * np.cos(ang_l)], [x_true[1], x_true[1] + ROBOT_RANGE * np.sin(ang_l)], 'g-', alpha=0.2)
    ax.plot([x_true[0], x_true[0] + ROBOT_RANGE * np.cos(ang_r)], [x_true[1], x_true[1] + ROBOT_RANGE * np.sin(ang_r)], 'g-', alpha=0.2)
    theta = np.linspace(ang_r, ang_l, 50)
    ax.plot(x_true[0] + ROBOT_RANGE * np.cos(theta), x_true[1] + ROBOT_RANGE * np.sin(theta), 'g-', alpha=0.2)


def run_slam_simulation(enable_da, show_realtime):
    random.seed(SEED)
    np.random.seed(SEED)
    NUM_STEPS = get_steps()

    P = 0.01 * np.eye(3)
    map_list = []

    x_true, x_ukf = get_initial_conditions(P)
    true_landmarks = generate_landmarks(NUM_LANDMARKS)
    true_history, path_history = [], []
    x_error, y_error, th_error = [], [], []

    # NEW: landmark error history {l_id: [(step, error), ...]}
    landmark_error_history = {}

    robot_model = DifferentialDrive()
    if enable_da:
        slam_system = UKF_SLAM_DA(robot_model, Q, R)
    else:
        slam_system = UKF_SLAM(robot_model, Q, R)

    if show_realtime:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        ax = None

    for t in range(NUM_STEPS):
        u_true = get_command(t, NUM_STEPS)
        x_true = robot_model.f(x_true.reshape(-1, 1), u_true)[:, 0]
        true_history.append((x_true[0], x_true[1]))

        u_noisy = {
            'r1': u_true['r1'] + np.random.normal(0, 0.02),
            't': u_true['t'] + np.random.normal(0, 0.05),
            'r2': u_true['r2'] + np.random.normal(0, 0.02)
        }

        # PREDICTION
        x_ukf, P = slam_system.predict(x_ukf, P, u_noisy)

        # OBSERVATION
        z_list = []
        for l_id, (lx, ly) in true_landmarks.items():
            z = robot_model.g(x_true.reshape(-1, 1), lx, ly)
            r_true, phi_true = z[0, 0], z[1, 0]
            if r_true < ROBOT_RANGE and abs(phi_true) < get_fov(ENABLED_FOV):
                obs = {'range': r_true + np.random.normal(0, 0.1),
                       'bearing': phi_true + np.random.normal(0, 0.05)}
                if enable_da:
                    obs['true_id'] = l_id
                else:
                    obs['id'] = l_id
                z_list.append(obs)

        # UPDATE
        if z_list:
            x_ukf, P, map_list = slam_system.update(x_ukf, P, z_list, map_list)

        path_history.append((x_ukf[0, 0], x_ukf[1, 0]))
        x_error.append(x_true[0] - x_ukf[0, 0])
        y_error.append(x_true[1] - x_ukf[1, 0])
        th_error.append(normalize_angle(x_true[2] - x_ukf[2, 0]))

        # NEW: track landmark errors over time
        for i, l_id in enumerate(map_list):
            if l_id > 0 and l_id in true_landmarks:
                ex = true_landmarks[l_id][0] - x_ukf[3 + 2 * i, 0]
                ey = true_landmarks[l_id][1] - x_ukf[3 + 2 * i + 1, 0]
                err = math.hypot(ex, ey)
                if l_id not in landmark_error_history:
                    landmark_error_history[l_id] = [(t, err)]
                else:
                    landmark_error_history[l_id].append((t, err))

        # PLOT REAL-TIME
        if show_realtime:
            draw_realtime_ax(ax, enable_da, true_landmarks, true_history, path_history, x_true, x_ukf, P, map_list)
            plt.pause(0.01)

    if show_realtime:
        plt.ioff()
        plt.close(fig) 

    # NOTE: now returns 10 elements (added landmark_error_history)
    return true_history, path_history, x_error, y_error, th_error, map_list, x_ukf, P, true_landmarks, landmark_error_history


def run_slam_comparison_simultaneous():
    random.seed(SEED)
    np.random.seed(SEED)
    NUM_STEPS = get_steps()

    P_noda = 0.01 * np.eye(3)
    P_da = 0.01 * np.eye(3)
    map_list_noda, map_list_da = [], []

    x_true_noda, x_ukf_noda = get_initial_conditions(P_noda)
    x_true_da = x_true_noda.copy()
    x_ukf_da = x_ukf_noda.copy()

    true_landmarks = generate_landmarks(NUM_LANDMARKS)
    
    true_history_noda, path_history_noda, x_error_noda, y_error_noda, th_error_noda = [], [], [], [], []
    true_history_da, path_history_da, x_error_da, y_error_da, th_error_da = [], [], [], [], []

    # NEW: landmark error histories for both systems
    landmark_error_history_noda = {}
    landmark_error_history_da = {}

    robot_noda = DifferentialDrive()
    robot_da = DifferentialDrive()
    slam_noda = UKF_SLAM(robot_noda, Q, R)
    slam_da = UKF_SLAM_DA(robot_da, Q, R)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Perfect IDs vs Data Association", fontsize=16)

    for t in range(NUM_STEPS):
        u_true = get_command(t, NUM_STEPS)
        
        x_true_noda = robot_noda.f(x_true_noda.reshape(-1, 1), u_true)[:, 0]
        x_true_da = robot_da.f(x_true_da.reshape(-1, 1), u_true)[:, 0]
        
        true_history_noda.append((x_true_noda[0], x_true_noda[1]))
        true_history_da.append((x_true_da[0], x_true_da[1]))

        u_noisy = {
            'r1': u_true['r1'] + np.random.normal(0, 0.02),
            't': u_true['t'] + np.random.normal(0, 0.05),
            'r2': u_true['r2'] + np.random.normal(0, 0.02)
        }

        x_ukf_noda, P_noda = slam_noda.predict(x_ukf_noda, P_noda, u_noisy)
        x_ukf_da, P_da = slam_da.predict(x_ukf_da, P_da, u_noisy)

        z_list_noda, z_list_da = [], []
        for l_id, (lx, ly) in true_landmarks.items():
            z = robot_noda.g(x_true_noda.reshape(-1, 1), lx, ly)
            r_true, phi_true = z[0, 0], z[1, 0]
            if r_true < ROBOT_RANGE and abs(phi_true) < get_fov(ENABLED_FOV):
                noise_r = np.random.normal(0, 0.1)
                noise_phi = np.random.normal(0, 0.05)
                
                obs_noda = {'range': r_true + noise_r, 'bearing': phi_true + noise_phi, 'id': l_id}
                obs_da = {'range': r_true + noise_r, 'bearing': phi_true + noise_phi, 'true_id': l_id}
                z_list_noda.append(obs_noda)
                z_list_da.append(obs_da)

        if z_list_noda:
            x_ukf_noda, P_noda, map_list_noda = slam_noda.update(x_ukf_noda, P_noda, z_list_noda, map_list_noda)
        if z_list_da:
            x_ukf_da, P_da, map_list_da = slam_da.update(x_ukf_da, P_da, z_list_da, map_list_da)

        path_history_noda.append((x_ukf_noda[0, 0], x_ukf_noda[1, 0]))
        x_error_noda.append(x_true_noda[0] - x_ukf_noda[0, 0])
        y_error_noda.append(x_true_noda[1] - x_ukf_noda[1, 0])
        th_error_noda.append(normalize_angle(x_true_noda[2] - x_ukf_noda[2, 0]))

        path_history_da.append((x_ukf_da[0, 0], x_ukf_da[1, 0]))
        x_error_da.append(x_true_da[0] - x_ukf_da[0, 0])
        y_error_da.append(x_true_da[1] - x_ukf_da[1, 0])
        th_error_da.append(normalize_angle(x_true_da[2] - x_ukf_da[2, 0]))

        # NEW: track landmark errors over time for both systems
        for i, l_id in enumerate(map_list_noda):
            if l_id > 0 and l_id in true_landmarks:
                ex = true_landmarks[l_id][0] - x_ukf_noda[3 + 2 * i, 0]
                ey = true_landmarks[l_id][1] - x_ukf_noda[3 + 2 * i + 1, 0]
                err = math.hypot(ex, ey)
                if l_id not in landmark_error_history_noda:
                    landmark_error_history_noda[l_id] = [(t, err)]
                else:
                    landmark_error_history_noda[l_id].append((t, err))

        for i, l_id in enumerate(map_list_da):
            if l_id > 0 and l_id in true_landmarks:
                ex = true_landmarks[l_id][0] - x_ukf_da[3 + 2 * i, 0]
                ey = true_landmarks[l_id][1] - x_ukf_da[3 + 2 * i + 1, 0]
                err = math.hypot(ex, ey)
                if l_id not in landmark_error_history_da:
                    landmark_error_history_da[l_id] = [(t, err)]
                else:
                    landmark_error_history_da[l_id].append((t, err))

        draw_realtime_ax(ax1, False, true_landmarks, true_history_noda, path_history_noda, x_true_noda, x_ukf_noda, P_noda, map_list_noda)
        draw_realtime_ax(ax2, True, true_landmarks, true_history_da, path_history_da, x_true_da, x_ukf_da, P_da, map_list_da)
        plt.pause(0.01)

    plt.ioff()
    plt.close(fig)

    res_noda = (true_history_noda, path_history_noda, x_error_noda, y_error_noda, th_error_noda, map_list_noda, x_ukf_noda, P_noda, true_landmarks, landmark_error_history_noda)
    res_da = (true_history_da, path_history_da, x_error_da, y_error_da, th_error_da, map_list_da, x_ukf_da, P_da, true_landmarks, landmark_error_history_da)
    
    return res_noda, res_da


# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_static_map(res, title_suffix, is_da):
    true_history, path_history, _, _, _, map_list, x_ukf, P, true_landmarks, _ = res
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"UKF-SLAM Results {title_suffix}", fontsize=16)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(RANGE_MIN - 1, RANGE_MAX + 1)
    ax.set_ylim(RANGE_MIN - 1, RANGE_MAX + 1)

    for l_id, (lx, ly) in true_landmarks.items():
        ax.plot(lx, ly, 'k*', markersize=12, label="True Landmark" if l_id == 1 else "")

    if len(true_history) > 1:
        px, py = zip(*true_history)
        ax.plot(px, py, 'g--', alpha=0.6, linewidth=2, label="True Path")
    if len(path_history) > 1:
        px, py = zip(*path_history)
        ax.plot(px, py, 'r--', alpha=0.6, linewidth=2, label="Estimated Path")

    ax.plot(true_history[-1][0], true_history[-1][1], 'go', markersize=10, label="True Robot")
    ax.plot(x_ukf[0, 0], x_ukf[1, 0], 'ro', markersize=10, label="Estimated Robot")
    ax.quiver(x_ukf[0, 0], x_ukf[1, 0], np.cos(x_ukf[2, 0]), np.sin(x_ukf[2, 0]), color='r', scale=15)

    for i, l_id in enumerate(map_list):
        lx, ly = x_ukf[3 + 2 * i, 0], x_ukf[3 + 2 * i + 1, 0]
        if is_da and l_id < 0:
            color, label = 'orange', "Est. Landmark (new)" if i == 0 else ""
        else:
            color, label = 'blue', "Est. Landmark" if i == 0 else ""
        ax.plot(lx, ly, 'o', color=color, markersize=6, label=label)
        ax.text(lx + 0.1, ly + 0.1, str(l_id), color=color, fontsize=8)
        plot_covariance_ellipse(lx, ly, P[3+2*i:3+2*i+2, 3+2*i:3+2*i+2], ax, color=color, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

def plot_pose_errors(res, title_suffix):
    _, _, x_error, y_error, th_error, _, _, _, _, _ = res
    steps = range(1, len(x_error) + 1)
    fig_err, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(steps, x_error, 'r-', linewidth=1.5)
    axes[0].set_ylabel("X Error (m)")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_title(f"Pose Errors {title_suffix}")

    axes[1].plot(steps, y_error, 'g-', linewidth=1.5)
    axes[1].set_ylabel("Y Error (m)")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[2].plot(steps, np.degrees(th_error), 'b-', linewidth=1.5)
    axes[2].set_ylabel("Theta Error (deg)")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_landmark_errors(res, title_suffix, is_da):
    _, _, _, _, _, map_list, x_ukf, _, true_landmarks, _ = res
    lm_ids, lm_errors, false_positives = [], [], 0
    for i, l_id in enumerate(map_list):
        if l_id > 0 and l_id in true_landmarks:
            error = math.hypot(true_landmarks[l_id][0] - x_ukf[3+2*i, 0], true_landmarks[l_id][1] - x_ukf[3+2*i+1, 0])
            lm_ids.append(l_id)
            lm_errors.append(error)
        elif is_da and l_id < 0:
            false_positives += 1

    if lm_ids:
        lm_ids_sorted, lm_errors_sorted = zip(*sorted(zip(lm_ids, lm_errors)))
        fig_lm, ax_lm = plt.subplots(figsize=(max(10, len(lm_ids) * 0.4), 5))
        colors = ['steelblue' if lid > 0 else 'orange' for lid in lm_ids_sorted]
        ax_lm.bar(range(len(lm_ids_sorted)), lm_errors_sorted, color=colors, edgecolor='black', linewidth=0.5)
        ax_lm.set_xticks(range(len(lm_ids_sorted)))
        ax_lm.set_xticklabels([f"LM {lid}" for lid in lm_ids_sorted], rotation=45, ha='right', fontsize=8)
        ax_lm.set_ylabel("Euclidean Error (m)")
        ax_lm.set_xlabel("Landmark ID")
        ax_lm.set_title(f"Landmark Position Error {title_suffix}", fontsize=14)
        ax_lm.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()


# ==========================================
# NEW: Landmark Error History Plot
# ==========================================

def plot_landmark_error_history(res, title_suffix, max_landmarks=None):
    """
    Plot the Euclidean error of each landmark over time (step by step).
    Shows how each landmark's position estimate improves (or not) as it is observed.

    Parameters
    ----------
    res          : simulation result tuple (10 elements)
    title_suffix : string label for the plot title
    max_landmarks: if set, only plot the first N landmark IDs (sorted); 
                   None means plot all
    """
    landmark_error_history = res[9]

    if not landmark_error_history:
        print(f"[plot_landmark_error_history] No data available for {title_suffix}.")
        return

    sorted_ids = sorted(landmark_error_history.keys())
    if max_landmarks is not None:
        sorted_ids = sorted_ids[:max_landmarks]

    n = len(sorted_ids)
    cmap = plt.cm.get_cmap('tab20', max(n, 1))

    fig, ax = plt.subplots(figsize=(13, 6))

    for idx, l_id in enumerate(sorted_ids):
        history = landmark_error_history[l_id]
        steps_lm, errors_lm = zip(*history)
        ax.plot(steps_lm, errors_lm,
                label=f"LM {l_id}",
                color=cmap(idx),
                linewidth=1.4,
                alpha=0.85)

    ax.set_title(f"Landmark Euclidean Error over Time {title_suffix}", fontsize=14)
    ax.set_xlabel("Step")
    ax.set_ylabel("Euclidean Error (m)")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legend: two columns when many landmarks
    ncol = 2 if n > 10 else 1
    ax.legend(loc='upper right', fontsize=7, ncol=ncol)

    plt.tight_layout()


def plot_comparison_landmark_error_history(res_da, res_noda, max_landmarks=None):
    """
    Side-by-side comparison of landmark error over time:
    left panel = Perfect IDs, right panel = Data Association.

    Only landmarks common to both runs are shown so the comparison is fair.
    """
    leh_noda = res_noda[9]
    leh_da   = res_da[9]

    common_ids = sorted(set(leh_noda.keys()) & set(leh_da.keys()))
    if not common_ids:
        print("[plot_comparison_landmark_error_history] No common landmarks to compare.")
        return

    if max_landmarks is not None:
        common_ids = common_ids[:max_landmarks]

    n = len(common_ids)
    cmap = plt.cm.get_cmap('tab20', max(n, 1))

    fig, (ax_noda, ax_da) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    fig.suptitle("Landmark Error over Time: Perfect IDs vs Data Association", fontsize=15, fontweight='bold')

    for idx, l_id in enumerate(common_ids):
        color = cmap(idx)
        label = f"LM {l_id}"

        # Perfect IDs
        steps_n, errors_n = zip(*leh_noda[l_id])
        ax_noda.plot(steps_n, errors_n, color=color, linewidth=1.4, alpha=0.85, label=label)

        # Data Association
        steps_d, errors_d = zip(*leh_da[l_id])
        ax_da.plot(steps_d, errors_d, color=color, linewidth=1.4, alpha=0.85, label=label)

    for ax, title in [(ax_noda, "Perfect IDs"), (ax_da, "Data Association")]:
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Step")
        ax.grid(True, linestyle='--', alpha=0.6)
        ncol = 2 if n > 10 else 1
        ax.legend(loc='upper left', fontsize=7, ncol=ncol)

    ax_noda.set_ylabel("Euclidean Error (m)")
    plt.tight_layout()


def plot_comparison_map(res_da, res_noda):
    true_history, path_da, _, _, _, _, _, _, true_landmarks, _ = res_da
    _, path_noda, _, _, _, _, _, _, _, _ = res_noda
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Data Association vs Perfect IDs", fontsize=16)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    for l_id, (lx, ly) in true_landmarks.items():
        ax.plot(lx, ly, 'k*', markersize=10, label="True Landmark" if l_id == 1 else "")
        
    px, py = zip(*true_history)
    ax.plot(px, py, 'g--', alpha=0.6, linewidth=2, label="True Path")
    
    px_n, py_n = zip(*path_noda)
    ax.plot(px_n, py_n, 'b--', alpha=0.7, linewidth=2, label="Estimated Path (Perfect IDs)")
    
    px_d, py_d = zip(*path_da)
    ax.plot(px_d, py_d, 'r--', alpha=0.7, linewidth=2, label="Estimated Path (Data Association)")
    
    ax.legend(loc='upper left')

def plot_comparison_pose_errors(res_da, res_noda):
    _, _, err_x_da, err_y_da, err_th_da, _, _, _, _, _ = res_da
    _, _, err_x_noda, err_y_noda, err_th_noda, _, _, _, _, _ = res_noda
    steps = range(1, len(err_x_da) + 1)
    
    fig_comp_err, axes_comp = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes_comp[0].plot(steps, err_x_da, 'r-', linewidth=1.5, label='Data Association')
    axes_comp[0].plot(steps, err_x_noda, 'b--', linewidth=1.5, label='Perfect IDs')
    axes_comp[0].set_ylabel("Error X (m)")
    axes_comp[0].legend(loc="upper right")
    axes_comp[0].grid(True, linestyle='--', alpha=0.7)

    axes_comp[1].plot(steps, err_y_da, 'r-', linewidth=1.5, label='Data Association')
    axes_comp[1].plot(steps, err_y_noda, 'b--', linewidth=1.5, label='Perfect IDs')
    axes_comp[1].set_ylabel("Error Y (m)")
    axes_comp[1].legend(loc="upper right")
    axes_comp[1].grid(True, linestyle='--', alpha=0.7)

    axes_comp[2].plot(steps, np.degrees(err_th_da), 'r-', linewidth=1.5, label='Data Association')
    axes_comp[2].plot(steps, np.degrees(err_th_noda), 'b--', linewidth=1.5, label='Perfect IDs')
    axes_comp[2].set_ylabel("Error Theta (deg)")
    axes_comp[2].set_xlabel("Step")
    axes_comp[2].legend(loc="upper right")
    axes_comp[2].grid(True, linestyle='--', alpha=0.7)
    fig_comp_err.suptitle("Comparison of Pose Errors", fontsize=16)
    plt.tight_layout()

def plot_comparison_landmark_errors(res_da, res_noda):
    _, _, _, _, _, map_list_da, x_ukf_da, _, true_lm_da, _ = res_da
    _, _, _, _, _, map_list_noda, x_ukf_noda, _, true_lm_noda, _ = res_noda

    dict_err_da = {l_id: math.hypot(true_lm_da[l_id][0] - x_ukf_da[3+2*i, 0], true_lm_da[l_id][1] - x_ukf_da[3+2*i+1, 0]) 
                   for i, l_id in enumerate(map_list_da) if l_id > 0 and l_id in true_lm_da}
    dict_err_noda = {l_id: math.hypot(true_lm_noda[l_id][0] - x_ukf_noda[3+2*i, 0], true_lm_noda[l_id][1] - x_ukf_noda[3+2*i+1, 0]) 
                     for i, l_id in enumerate(map_list_noda) if l_id > 0 and l_id in true_lm_noda}

    common_ids = sorted(list(set(dict_err_da.keys()) | set(dict_err_noda.keys())))
    vals_da = [dict_err_da.get(lid, 0) for lid in common_ids]
    vals_noda = [dict_err_noda.get(lid, 0) for lid in common_ids]

    fig_comp_lm, ax_comp_lm = plt.subplots(figsize=(max(12, len(common_ids) * 0.4), 6))
    x_indices = np.arange(len(common_ids))
    width = 0.35

    ax_comp_lm.bar(x_indices - width / 2, vals_da, width, label='Data Association', color='indianred', edgecolor='black')
    ax_comp_lm.bar(x_indices + width / 2, vals_noda, width, label='Perfect IDs', color='steelblue', edgecolor='black')
    ax_comp_lm.set_xticks(x_indices)
    ax_comp_lm.set_xticklabels([f"LM {lid}" for lid in common_ids], rotation=45, ha='right', fontsize=9)
    ax_comp_lm.set_ylabel("Euclidean Error (m)")
    ax_comp_lm.set_title("Comparison of Landmark Positioning Errors", fontsize=16)
    ax_comp_lm.legend()
    ax_comp_lm.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()


# ==========================================
# MAIN
# ==========================================

def main():
    if MODE_PERFECT_ID_ONLY:
        res = run_slam_simulation(enable_da=False, show_realtime=ENABLE_REALTIME)
        if ENABLE_RESULTS:
            plot_static_map(res, "(Perfect IDs)", False)
        if ENABLE_ERRORS:
            plot_pose_errors(res, "(Perfect IDs)")
        if ENABLE_LANDMARKERROR:
            plot_landmark_errors(res, "(Perfect IDs)", False)
        if ENABLE_LANDMARK_ERROR_HISTORY:
            plot_landmark_error_history(res, "(Perfect IDs)")

    elif MODE_DATA_ASSOCIATION_ONLY:
        res = run_slam_simulation(enable_da=True, show_realtime=ENABLE_REALTIME)
        if ENABLE_RESULTS:
            plot_static_map(res, "(Data Association)", True)
        if ENABLE_ERRORS:
            plot_pose_errors(res, "(Data Association)")
        if ENABLE_LANDMARKERROR:
            plot_landmark_errors(res, "(Data Association)", True)
        if ENABLE_LANDMARK_ERROR_HISTORY:
            plot_landmark_error_history(res, "(Data Association)")

    elif MODE_COMPARISON:
        if ENABLE_REALTIME:
            res_noda, res_da = run_slam_comparison_simultaneous()
        else:
            res_noda = run_slam_simulation(enable_da=False, show_realtime=False)
            res_da = run_slam_simulation(enable_da=True, show_realtime=False)

        if ENABLE_RESULTS:
            plot_comparison_map(res_da, res_noda)
        if ENABLE_ERRORS:
            plot_comparison_pose_errors(res_da, res_noda)
        if ENABLE_LANDMARKERROR:
            plot_comparison_landmark_errors(res_da, res_noda)
        if ENABLE_LANDMARK_ERROR_HISTORY:
            plot_comparison_landmark_error_history(res_da, res_noda)

    plt.show()

if __name__ == '__main__':
    main()