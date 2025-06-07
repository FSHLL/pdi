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

def median_filter(f, size=3):
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

                g[z, x, y] = np.median(neighborhood)
    return g

def anisotropic_diffusion(f, num_iter=10, kappa=30, gamma=0.1):
    g = f.astype(np.float32).copy()
    for _ in range(num_iter):
        # Gradientes en las 6 direcciones
        grad_n = np.roll(g, -1, axis=0) - g
        grad_s = np.roll(g, 1, axis=0) - g
        grad_e = np.roll(g, -1, axis=1) - g
        grad_w = np.roll(g, 1, axis=1) - g
        grad_up = np.roll(g, -1, axis=2) - g
        grad_down = np.roll(g, 1, axis=2) - g

        # Coeficientes de difusión (función de la norma del gradiente)
        c_n = np.exp(-(grad_n / kappa) ** 2)
        c_s = np.exp(-(grad_s / kappa) ** 2)
        c_e = np.exp(-(grad_e / kappa) ** 2)
        c_w = np.exp(-(grad_w / kappa) ** 2)
        c_up = np.exp(-(grad_up / kappa) ** 2)
        c_down = np.exp(-(grad_down / kappa) ** 2)

        # Actualización
        g += gamma * (
            c_n * grad_n + c_s * grad_s +
            c_e * grad_e + c_w * grad_w +
            c_up * grad_up + c_down * grad_down
        )
    return g

def non_local_means_filter(f, patch_size=1, search_size=3, h=0.1):
    f = f.astype(np.float32)
    g = np.zeros_like(f)
    pad_width = search_size + patch_size
    f_padded = np.pad(f, pad_width, mode='reflect')

    Z, X, Y = f.shape

    for z in range(Z):
        for x in range(X):
            for y in range(Y):
                zc, xc, yc = z + pad_width, x + pad_width, y + pad_width
                patch_ref = f_padded[
                    zc - patch_size: zc + patch_size + 1,
                    xc - patch_size: xc + patch_size + 1,
                    yc - patch_size: yc + patch_size + 1
                ]
                weights = []
                patches = []
                for dz in range(-search_size, search_size + 1):
                    for dx in range(-search_size, search_size + 1):
                        for dy in range(-search_size, search_size + 1):
                            zn, xn, yn = zc + dz, xc + dx, yc + dy
                            patch = f_padded[
                                zn - patch_size: zn + patch_size + 1,
                                xn - patch_size: xn + patch_size + 1,
                                yn - patch_size: yn + patch_size + 1
                            ]
                            dist2 = np.sum((patch - patch_ref) ** 2)
                            w = np.exp(-dist2 / (h ** 2))
                            weights.append(w)
                            patches.append(f_padded[zn, xn, yn])
                weights = np.array(weights)
                patches = np.array(patches)
                g[z, x, y] = np.sum(weights * patches) / np.sum(weights)
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

    for _ in range(max_iter):
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