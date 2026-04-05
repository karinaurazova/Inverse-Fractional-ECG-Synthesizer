#!/usr/bin/env python3

import numpy as np
from scipy.io import wavfile
from scipy import signal
import scipy.special as sp
from scipy.linalg import toeplitz, lstsq
import argparse
import warnings

warnings.filterwarnings("ignore")


class InverseFractionalECGSynthesizer:
    def __init__(self, fs=1000, memory_length=100):
        self.fs = fs
        self.M = memory_length
        self.coeff_cache = {}

    def get_grunwald_letnikov_coeffs(self, alpha):
        if alpha in self.coeff_cache:
            return self.coeff_cache[alpha]

        coeffs = np.zeros(self.M + 1)
        coeffs[0] = 1.0
        for m in range(1, self.M + 1):
            numer = sp.gamma(alpha + 1)
            denom = sp.gamma(m + 1) * sp.gamma(alpha - m + 1)
            coeffs[m] = ((-1) ** m) * numer / denom

        coeffs = coeffs / np.sum(np.abs(coeffs))
        self.coeff_cache[alpha] = coeffs
        return coeffs

    def fractional_derivative_matrix(self, N, alpha):
        coeffs = self.get_grunwald_letnikov_coeffs(alpha)
        c = np.zeros(N)
        c[0] = coeffs[0]
        c[1:min(self.M, N - 1) + 1] = coeffs[1:min(self.M, N - 1) + 1]

        r = np.zeros(N)
        r[0] = coeffs[0]

        T = toeplitz(c, r)
        return T

    def solve_inverse_fractional(self, target_signal, alpha, regularization=1e-6):
        N = len(target_signal)
        T = self.fractional_derivative_matrix(N, alpha)

        T_h = T.conj().T
        lhs = T_h @ T + regularization * np.eye(N)
        rhs = T_h @ target_signal

        x = np.linalg.solve(lhs, rhs)
        return x

    def generate_ecg_from_peaks(self, duration=10, hr_bpm=75,
                                r_alpha=1.8, p_alpha=0.7, t_alpha=0.9,
                                r_amplitude=1.0, p_amplitude=0.15, t_amplitude=0.3,
                                q_amplitude=-0.05, s_amplitude=-0.1,
                                p_offset=-0.18, q_offset=-0.05, s_offset=0.05, t_offset=0.25,
                                noise_level=0.02, drift_amplitude=0.05, drift_freq=0.15):
        t = np.arange(0, duration, 1.0 / self.fs)
        N = len(t)

        mean_rr = 60.0 / hr_bpm
        rr_intervals = np.random.normal(mean_rr, mean_rr * 0.03, int(duration / mean_rr) + 2)
        rr_intervals = np.maximum(rr_intervals, 0.4)
        r_positions = np.cumsum(rr_intervals)
        r_positions = r_positions[r_positions < duration]
        if len(r_positions) == 0:
            raise ValueError("No R peaks in given duration")

        r_idx = (r_positions * self.fs).astype(int)
        r_idx = r_idx[r_idx < N]

        target_qrs = np.zeros(N, dtype=complex)
        sigma_r_peak = int(0.01 * self.fs)
        for idx in r_idx:
            left = max(0, idx - 3 * sigma_r_peak)
            right = min(N, idx + 3 * sigma_r_peak)
            gauss = np.exp(-0.5 * ((np.arange(left, right) - idx) / sigma_r_peak) ** 2)
            target_qrs[left:right] += r_amplitude * gauss

        target_p = np.zeros(N, dtype=complex)
        target_t = np.zeros(N, dtype=complex)
        sigma_pt_peak = int(0.02 * self.fs)

        for idx in r_idx:
            p_center = idx + int(p_offset * self.fs)
            if 0 <= p_center < N:
                left = max(0, p_center - 3 * sigma_pt_peak)
                right = min(N, p_center + 3 * sigma_pt_peak)
                gauss = np.exp(-0.5 * ((np.arange(left, right) - p_center) / sigma_pt_peak) ** 2)
                target_p[left:right] += p_amplitude * gauss

            t_center = idx + int(t_offset * self.fs)
            if 0 <= t_center < N:
                left = max(0, t_center - 3 * sigma_pt_peak)
                right = min(N, t_center + 3 * sigma_pt_peak)
                gauss = np.exp(-0.5 * ((np.arange(left, right) - t_center) / sigma_pt_peak) ** 2)
                target_t[left:right] += t_amplitude * gauss

        target_q = np.zeros(N, dtype=complex)
        target_s = np.zeros(N, dtype=complex)
        sigma_qs_peak = int(0.008 * self.fs)

        for idx in r_idx:
            q_center = idx + int(q_offset * self.fs)
            if 0 <= q_center < N:
                left = max(0, q_center - 3 * sigma_qs_peak)
                right = min(N, q_center + 3 * sigma_qs_peak)
                gauss = np.exp(-0.5 * ((np.arange(left, right) - q_center) / sigma_qs_peak) ** 2)
                target_q[left:right] += q_amplitude * gauss

            s_center = idx + int(s_offset * self.fs)
            if 0 <= s_center < N:
                left = max(0, s_center - 3 * sigma_qs_peak)
                right = min(N, s_center + 3 * sigma_qs_peak)
                gauss = np.exp(-0.5 * ((np.arange(left, right) - s_center) / sigma_qs_peak) ** 2)
                target_s[left:right] += s_amplitude * gauss

        S_R = self.solve_inverse_fractional(target_qrs, r_alpha, regularization=1e-5)
        S_P = self.solve_inverse_fractional(target_p, p_alpha, regularization=1e-5)
        S_T = self.solve_inverse_fractional(target_t, t_alpha, regularization=1e-5)
        S_Q = self.solve_inverse_fractional(target_q, r_alpha, regularization=1e-5)
        S_S = self.solve_inverse_fractional(target_s, r_alpha, regularization=1e-5)

        ecg = np.real(S_R + S_P + S_T + S_Q + S_S)

        drift = drift_amplitude * np.sin(2 * np.pi * drift_freq * t)
        ecg += drift

        if noise_level > 0:
            ecg += np.random.normal(0, noise_level, N)

        ecg = ecg - np.mean(ecg)

        max_amp = np.max(np.abs(ecg))
        if max_amp > 0:
            ecg = ecg / max_amp * 0.9

        return ecg, t, {'r_positions': r_positions}


def save_wav(filename, signal, fs):
    signal_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(filename, fs, signal_int16)


def main():
    parser = argparse.ArgumentParser(description="Synthetic ECG generator using inverse fractional derivative")
    parser.add_argument("--output", "-o", type=str, default="inverse_fractional_ecg.wav")
    parser.add_argument("--duration", "-d", type=float, default=10.0)
    parser.add_argument("--fs", type=int, default=1000)
    parser.add_argument("--hr", type=float, default=75.0)
    parser.add_argument("--r_alpha", type=float, default=1.8)
    parser.add_argument("--p_alpha", type=float, default=0.7)
    parser.add_argument("--t_alpha", type=float, default=0.9)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--drift_amp", type=float, default=0.05)
    parser.add_argument("--drift_freq", type=float, default=0.15)
    args = parser.parse_args()

    synthesizer = InverseFractionalECGSynthesizer(fs=args.fs)
    ecg, t, info = synthesizer.generate_ecg_from_peaks(
        duration=args.duration,
        hr_bpm=args.hr,
        r_alpha=args.r_alpha,
        p_alpha=args.p_alpha,
        t_alpha=args.t_alpha,
        noise_level=args.noise,
        drift_amplitude=args.drift_amp,
        drift_freq=args.drift_freq
    )
    save_wav(args.output, ecg, args.fs)
    print(f"Generated {args.output}, R peaks: {len(info['r_positions'])}")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(t, ecg)
        for r in info['r_positions']:
            plt.axvline(r, color='red', linestyle='--', alpha=0.5)
        plt.title("Synthetic ECG (inverse fractional derivative)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()