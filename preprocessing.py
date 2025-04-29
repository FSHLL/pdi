import numpy as np
from scipy.ndimage import gaussian_filter

def mean_filter(f, size=3):
    g = np.zeros_like(f, dtype=np.float32)

    radius = size // 3

    for z in range(f.shape[0]):
        for x in range(f.shape[1]):
            for y in range(f.shape[2]):
                z_min = max(0, z - radius)
                z_max = min(f.shape[0], z + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(f.shape[1], x + radius + 1)
                y_min = max(0, y - radius)
                y_max = min(f.shape[2], y + radius + 1)

                neighborhood = f[z_min:z_max, x_min:x_max, y_min:y_max]

                g[z, x, y] = np.mean(neighborhood)

    return g

def normalize(f):
    f_min = np.min(f)
    f_max = np.max(f)

    if f_max - f_min == 0:
        return np.zeros_like(f, dtype=np.float32)

    g = (f - f_min) / (f_max - f_min)
    return g

def bias_field_correction(f, sigma=50):
    bias_field = gaussian_filter(f, sigma)

    bias_field[bias_field == 0] = 1

    g = f / bias_field

    g = (g - np.min(g)) / (np.max(g) - np.min(g))

    return g

def n3(f, max_iter=50, tol=1e-5, sigma=50):
    f = (f - np.min(f)) / (np.max(f) - np.min(f))

    bias_field = np.ones_like(f, dtype=np.float32)

    for iteration in range(max_iter):
        corrected_image = f / bias_field

        bias_field_update = gaussian_filter(corrected_image, sigma=sigma)

        new_bias_field = bias_field * bias_field_update

        if np.linalg.norm(new_bias_field - bias_field) < tol:
            print('finish')
            break

        bias_field = new_bias_field

    g = f / bias_field

    g = (g - np.min(g)) / (np.max(g) - np.min(g))

    return g

def gradient_distribution(f):
    gx = np.gradient(f, axis=0)
    gy = np.gradient(f, axis=1)
    gz = np.gradient(f, axis=2)

    magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
    return magnitude

def isotropic_diffusion(f, num_iter=10, t=0.12):
    g = f.copy()

    for _ in range(num_iter):
        grad_n = np.roll(g, -1, axis=0) - g
        grad_s = np.roll(g, 1, axis=0) - g
        grad_e = np.roll(g, -1, axis=1) - g
        grad_w = np.roll(g, 1, axis=1) - g

        grad_up = np.roll(g, -1, axis=2) - g
        grad_down = np.roll(g, 1, axis=2) - g

        g = g + t * (grad_n + grad_s + grad_e + grad_w + grad_up + grad_down)

    return g