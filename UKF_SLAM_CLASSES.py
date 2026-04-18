import numpy as np
import scipy.linalg
from utils.angle import normalize_angle

class UKF_SLAM:
    def __init__(self, robot_model, Q_noise, R_noise, alpha=1e-2, beta=2, gamma=0):
        self.robot = robot_model
        self.Q = Q_noise
        self.R = R_noise
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_weights(self, n):
        c = (self.alpha ** 2) * (n + self.gamma)
        wm = np.zeros(2 * n + 1)
        wc = np.zeros(2 * n + 1)
        wm[0] = 1 - n / c
        wc[0] = (2 - self.alpha ** 2 + self.beta) - n / c
        for i in range(1, 2 * n + 1):
            wm[i] = 1 / (2 * c)
            wc[i] = 1 / (2 * c)
        return wm, wc, c

    def compute_sigma_points(self, x_hat, P):
        n = len(x_hat)
        c = (self.alpha ** 2) * (n + self.gamma)
        sqrtCP = scipy.linalg.sqrtm(c * P)
        return np.hstack([
            x_hat,
            x_hat + sqrtCP,
            x_hat - sqrtCP
        ])

    def predict(self, xkk, Pkk, u):
        n = len(xkk)
        Xikk = self.compute_sigma_points(xkk, Pkk)
        Xikk[2, :] = normalize_angle(Xikk[2, :])

        Psik1k = self.robot.f(Xikk, u)

        wm, wc, c = self.get_weights(n)

        xk1k = np.sum(Psik1k * wm, axis=1, keepdims=True)
        sum_cos = np.sum(np.cos(Psik1k[2, :]) * wm)
        sum_sin = np.sum(np.sin(Psik1k[2, :]) * wm)
        xk1k[2, 0] = normalize_angle(np.arctan2(sum_sin, sum_cos))

        dx = Psik1k - xk1k
        dx[2, :] = normalize_angle(dx[2, :])

        W = np.zeros_like(Pkk)
        W[0:3, 0:3] = self.R

        Pk1k = W + ((dx * wc) @ dx.T)
        return xk1k, Pk1k

    def add_landmark_to_map(self, x_hat, P_hat, z, map_list):
        n = len(x_hat)
        n_aug = n + 2

        x_aug = np.vstack((x_hat, [[0.0], [0.0]]))
        P_aug = np.zeros((n_aug, n_aug))
        P_aug[:n, :n] = P_hat
        P_aug[n:, n:] = self.Q

        wm, wc, c = self.get_weights(n_aug)
        Xi_hat = self.compute_sigma_points(x_aug, P_aug)

        x, y, theta = Xi_hat[0, :], Xi_hat[1, :], Xi_hat[2, :]
        r, phi = Xi_hat[n, :], Xi_hat[n + 1, :]
        r_z, phi_z = z['range'], z['bearing']

        mx = x + (r_z + r) * np.cos(theta + phi_z + phi)
        my = y + (r_z + r) * np.sin(theta + phi_z + phi)
        m = np.vstack((mx, my))

        m_hat = np.sum(m * wm, axis=1, keepdims=True)

        dl = m - m_hat
        Pl = (dl * wc) @ dl.T

        dx = Xi_hat[:n, :] - x_hat
        dx[2, :] = normalize_angle(dx[2, :])
        Pxl = (dx * wc) @ dl.T

        x_ret = np.vstack((x_hat, m_hat))
        P_ret = np.zeros((n + 2, n + 2))
        P_ret[:n, :n] = P_hat
        P_ret[:n, n:] = Pxl
        P_ret[n:, :n] = Pxl.T
        P_ret[n:, n:] = Pl

        map_list.append(z['id'])
        return x_ret, P_ret, map_list

    def update(self, x_hat, P_hat, z_list, map_list):
        """Update con ID Perfetti."""
        for z in z_list:
            if z['id'] not in map_list:
                x_hat, P_hat, map_list = self.add_landmark_to_map(x_hat, P_hat, z, map_list)
                continue

            n = len(x_hat)
            wm, wc, c = self.get_weights(n)

            Xikk1 = self.compute_sigma_points(x_hat, P_hat)
            Xikk1[2, :] = normalize_angle(Xikk1[2, :])

            idx = map_list.index(z['id'])
            landmarkXs = Xikk1[3 + 2 * idx, :]
            landmarkYs = Xikk1[3 + 2 * idx + 1, :]
            zkk1 = self.robot.g(Xikk1, landmarkXs, landmarkYs)

            ykk1 = np.sum(zkk1 * wm, axis=1, keepdims=True)
            ykk1[1, 0] = normalize_angle(np.arctan2(
                np.sum(np.sin(zkk1[1, :]) * wm),
                np.sum(np.cos(zkk1[1, :]) * wm)
            ))

            dy = zkk1 - ykk1
            dy[1, :] = normalize_angle(dy[1, :])
            Py = self.Q + ((dy * wc) @ dy.T)

            dx = Xikk1 - x_hat
            dx[2, :] = normalize_angle(dx[2, :])
            Pxy = 1 / (2 * self.alpha ** 2 * (n + self.gamma)) * (dx @ dy.T)

            Lk = Pxy @ np.linalg.inv(Py)

            yk = np.array([[z['range']], [z['bearing']]])
            delta_y = yk - ykk1
            delta_y[1, 0] = normalize_angle(delta_y[1, 0])

            x_hat = x_hat + Lk @ delta_y
            P_hat = P_hat - Lk @ Py @ Lk.T
            x_hat[2, 0] = normalize_angle(x_hat[2, 0])

        return x_hat, P_hat, map_list


class UKF_SLAM_DA(UKF_SLAM):
    def __init__(self, robot_model, Q_noise, R_noise, gate_threshold=5.991):
        super().__init__(robot_model, Q_noise, R_noise)
        self.gate_threshold = gate_threshold

    def compute_mahalanobis_distance(self, z, landmark_idx, x_hat, P_hat):
        n = len(x_hat)
        wm, wc, c = self.get_weights(n)
        
        Xikk1 = self.compute_sigma_points(x_hat, P_hat)
        Xikk1[2, :] = normalize_angle(Xikk1[2, :])

        landmarkXs = Xikk1[3 + 2 * landmark_idx, :]
        landmarkYs = Xikk1[3 + 2 * landmark_idx + 1, :]
        zkk1 = self.robot.g(Xikk1, landmarkXs, landmarkYs)

        ykk1 = np.sum(zkk1 * wm, axis=1, keepdims=True)
        ykk1[1, 0] = normalize_angle(np.arctan2(
            np.sum(np.sin(zkk1[1, :]) * wm),
            np.sum(np.cos(zkk1[1, :]) * wm)
        ))

        yk = np.array([[z['range']], [z['bearing']]])
        innovation = yk - ykk1
        innovation[1, 0] = normalize_angle(innovation[1, 0])

        dy = zkk1 - ykk1
        dy[1, :] = normalize_angle(dy[1, :])
        S = self.Q + ((dy * wc) @ dy.T)

        distance = innovation.T @ np.linalg.inv(S) @ innovation
        return distance[0, 0], innovation, S

    def associate_observations(self, z_list, map_list, x_hat, P_hat):
        associations = {}
        new_observations = []
        used_landmarks = set()

        for i, z in enumerate(z_list):
            best_idx = None
            best_distance = self.gate_threshold

            for j, lm_id in enumerate(map_list):
                if j in used_landmarks:
                    continue
                distance, _, _ = self.compute_mahalanobis_distance(z, j, x_hat, P_hat)
                if distance < best_distance:
                    best_distance = distance
                    best_idx = j

            if best_idx is not None:
                associations[i] = best_idx
                used_landmarks.add(best_idx)
            else:
                new_observations.append(z)

        return associations, new_observations

    def update(self, x_hat, P_hat, z_list, map_list):
        if len(map_list) == 0:
            # Primo step
            for z in z_list:
                z['id'] = z.get('true_id', -len(map_list) - 1)
                x_hat, P_hat, map_list = self.add_landmark_to_map(x_hat, P_hat, z, map_list)
            return x_hat, P_hat, map_list

        associations, new_observations = self.associate_observations(z_list, map_list, x_hat, P_hat)
        processed_landmarks = set()

        for obs_idx, lm_idx in associations.items():
            if lm_idx in processed_landmarks:
                continue

            z = z_list[obs_idx]
            z['id'] = z.get('true_id', map_list[lm_idx])

            n = len(x_hat)
            wm, wc, c = self.get_weights(n)

            Xikk1 = self.compute_sigma_points(x_hat, P_hat)
            Xikk1[2, :] = normalize_angle(Xikk1[2, :])

            landmarkXs = Xikk1[3 + 2 * lm_idx, :]
            landmarkYs = Xikk1[3 + 2 * lm_idx + 1, :]
            zkk1 = self.robot.g(Xikk1, landmarkXs, landmarkYs)

            ykk1 = np.sum(zkk1 * wm, axis=1, keepdims=True)
            ykk1[1, 0] = normalize_angle(np.arctan2(
                np.sum(np.sin(zkk1[1, :]) * wm),
                np.sum(np.cos(zkk1[1, :]) * wm)
            ))

            dy = zkk1 - ykk1
            dy[1, :] = normalize_angle(dy[1, :])
            Py = self.Q + ((dy * wc) @ dy.T)

            dx = Xikk1 - x_hat
            dx[2, :] = normalize_angle(dx[2, :])
            Pxy = 1 / (2 * self.alpha ** 2 * (n + self.gamma)) * (dx @ dy.T)

            Lk = Pxy @ np.linalg.inv(Py)

            yk = np.array([[z['range']], [z['bearing']]])
            delta_y = yk - ykk1
            delta_y[1, 0] = normalize_angle(delta_y[1, 0])

            x_hat = x_hat + Lk @ delta_y
            P_hat = P_hat - Lk @ Py @ Lk.T
            x_hat[2, 0] = normalize_angle(x_hat[2, 0])

            processed_landmarks.add(lm_idx)

        for z in new_observations:
            z['id'] = z.get('true_id', -len(map_list) - 1)
            x_hat, P_hat, map_list = self.add_landmark_to_map(x_hat, P_hat, z, map_list)

        return x_hat, P_hat, map_list

