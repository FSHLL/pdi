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

def region_growing(f, seed, delta):
    g = np.zeros_like(f, dtype=np.uint8)
    z0, x0, y0 = seed
    cluster = [(z0, x0, y0)]
    g[z0, x0, y0] = 1 
    mcluster = f[z0, x0, y0]

    while cluster:
        new_cluster = []
        for z, x, y in cluster:
            neighbors = [
                (z-1, x, y), (z+1, x, y),
                (z, x-1, y), (z, x+1, y),
                (z, x, y-1), (z, x, y+1)
            ]
            for zn, xn, yn in neighbors:
                if (
                    0 <= zn < f.shape[0] and
                    0 <= xn < f.shape[1] and
                    0 <= yn < f.shape[2] and
                    g[zn, xn, yn] == 0
                ):
                    if abs(f[zn, xn, yn] - mcluster) < delta:
                        g[zn, xn, yn] = 1
                        new_cluster.append((zn, xn, yn))
        
        if not new_cluster:
            break
        
        cluster = new_cluster
        mcluster = np.mean([f[z, x, y] for z, x, y in cluster])

    return g

def kmeans_segmentation(f, k, max_iter=100, tol=0.5):
    flat_f = f.flatten()
    n_voxels = flat_f.shape[0]

    centroids = np.random.choice(flat_f, k, replace=False)

    labels = np.zeros(n_voxels, dtype=np.int32)

    for _ in range(max_iter):
        for i in range(n_voxels):
            distances = np.abs(flat_f[i] - centroids)
            labels[i] = np.argmin(distances)

        new_centroids = []

        for cluster in range(k):
            cluster_points = flat_f[labels == cluster]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean())
            else:
                new_centroids.append(centroids[cluster])

        new_centroids = np.array(new_centroids)

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    g = labels.reshape(f.shape)

    return g, centroids