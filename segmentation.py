import numpy as np

def isodata_thresholding(f, tau, delta_tau):
    t = 0  # Iteración inicial
    while True:
        g = f > tau

        foreground_pixels = f[g == 1]
        background_pixels = f[g == 0]

        mforeground = np.mean(foreground_pixels) if len(
            foreground_pixels) > 0 else 0
        mbackground = np.mean(background_pixels) if len(
            background_pixels) > 0 else 0

        tau_new = 0.5 * (mforeground + mbackground)

        if abs(tau_new - tau) < delta_tau:
            break

        tau = tau_new
        t += 1

    return g, tau
