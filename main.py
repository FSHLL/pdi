import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons
import tkinter
import segmentation
import preprocessing
from tkinter import filedialog
from tools import load_nifti, save_zip, load_zip

window = tkinter.Tk()
window.withdraw()

image_path = None
current_slice = 0
points = []
points_foreground = []
region_growing_enabled = False

fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)

ax_load_btn = plt.axes([0.05, 0.85, 0.2, 0.075])
load_btn = Button(ax_load_btn, 'Cargar NIfTI')

ax_load_zip_btn = plt.axes([0.05, 0.75, 0.2, 0.075])
load_zip_btn = Button(ax_load_zip_btn, 'Cargar ZIP')

ax_save_btn = plt.axes([0.05, 0.65, 0.2, 0.075])
save_btn = Button(ax_save_btn, 'Guardar ZIP')

segmentation_algorithms = ['Isodata', "Region Growing", "K-means", "Laplacian"]
selected_algorithm = [segmentation_algorithms[0]]

ax_radio = plt.axes([0.05, 0.4, 0.2, 0.2], frameon=True)
radio_buttons = RadioButtons(ax_radio, segmentation_algorithms)

ax_apply_btn = plt.axes([0.05, 0.3, 0.2, 0.075])
apply_btn = Button(ax_apply_btn, 'Aplicar')

preprocessing_algorithms = ['Mean filter', 'Normalization', 'Bias Fields', 'N3', 'Isotropic Diffusion']
selected_preprocessing = [preprocessing_algorithms[0]]

ax_preprocessing_radio = plt.axes([0.05, 0.1, 0.2, 0.2], frameon=True)
preprocessing_radio_buttons = RadioButtons(ax_preprocessing_radio, preprocessing_algorithms)

ax_preprocessing_btn = plt.axes([0.05, 0.02, 0.2, 0.075])
preprocessing_btn = Button(ax_preprocessing_btn, 'Aplicar Preprocesamiento')

def open_file(event):
    global image, image_path, current_slice
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    if file_path:
        image_path = file_path
        image = load_nifti(file_path, current_slice, ax, fig)
        current_slice = image.shape[2] // 2
        clear_drawing()
        ax.imshow(image[..., current_slice])
        fig.canvas.draw()
        check_draw()

def update_image(event):
    global current_slice
    print(event)
    if event.button == 'up':
        current_slice += 1
    elif event.button == 'down':
        current_slice -= 1
    if current_slice >= image.shape[2]:
        current_slice = 0
    clear_drawing()
    ax.imshow(image[..., current_slice])
    fig.canvas.draw()
    check_draw()

def draw_or_erase(event):
    if event.inaxes == ax and event.button == 1:
        draw_point(event)
    elif event.inaxes == ax and event.button == 3:
        draw_point_foreground(event)

def draw_continuous(event):
    if event.inaxes == ax and event.button == 1:
        draw_point(event)

def clear_drawing():
    ax.cla()
    ax.imshow(image[..., current_slice])
    fig.canvas.draw()

def draw_point(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        points.append((current_slice, (x, y)))
        ax.plot(x, y, 'ro', markersize=5)
        fig.canvas.draw()

def draw_point_foreground(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        points_foreground.append((current_slice, (x, y)))
        ax.plot(x, y, 'ro', markersize=5, color='blue')
        fig.canvas.draw()

def erase_point(event):
    global points
    if not points:
        return

    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return

    slice_points = [(idx, p) for idx, p in enumerate(points) if p[0] == current_slice]

    if not slice_points or event.button != 3:
        return

    min_dist = float('inf')
    closest_idx = None
    for idx, (_, (px, py)) in slice_points:
        dist = (px - x) ** 2 + (py - y) ** 2  # Distancia euclidiana al cuadrado
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx

    if closest_idx is not None:
        del points[closest_idx]
        update_image(event)

def check_draw():
    slice_points = [point for point in points if point[0] == current_slice]
    for point in slice_points:
        x, y = point[1]
        ax.plot(x, y, 'ro', markersize=5)
    fig.canvas.draw()

def clear_points():
    global points, points_foreground
    points = [pt for pt in points if pt[0] != current_slice]
    points_foreground = [pt for pt in points_foreground if pt[0] != current_slice]

def save_zip_event(event):
    try:
        zip_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip")],
            title="Guardar ZIP"
        )
        if zip_path:
            save_zip(image_path, points, zip_path)
            print(f"Archivo guardado en {zip_path}")
    except ValueError as e:
        print(e)

def load_zip_event(event):
    global image_path, points, current_slice
    zip_path = filedialog.askopenfilename(
        filetypes=[("ZIP files", "*.zip")],
        title="Seleccionar archivo ZIP"
    )
    if zip_path:
        nifti_file, loaded_points = load_zip(zip_path)
        if nifti_file:
            image_path = nifti_file
            image = load_nifti(nifti_file, current_slice, ax, fig)
            current_slice = image.shape[2] // 2
            clear_drawing()
            ax.imshow(image[..., current_slice])
            fig.canvas.draw()
            check_draw()
        points = loaded_points
        print(f"Archivo ZIP {zip_path} cargado correctamente")

def on_algorithm_selected(label):
    """Actualizar el algoritmo seleccionado."""
    selected_algorithm[0] = label
    print(f"Algoritmo seleccionado: {label}")

def apply_segmentation(event):
    global image, region_growing_enabled
    if image_path is None:
        print("No hay imagen.")
        return

    algorithm = selected_algorithm[0]
    print(f"Aplicando {algorithm}...")

    if algorithm == "Isodata":
        image, t = segmentation.isodata_thresholding(image, 0.5, 0.1)
    elif algorithm == "Region Growing":
        seed_point = (image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2)
        image = segmentation.region_growing(image, seed_point, 80)
        region_growing_enabled = True
    elif algorithm == "K-means":
        k = 3
        image, centroids = segmentation.kmeans_segmentation(image, k)
        print(f"Centroides finales: {centroids}")
    elif algorithm == "Laplacian":
        slice_points = [point for point in points if point[0] == current_slice]
        slice_points = [point for _, point in slice_points]
        slice_points = [(int(round(y)), int(round(x))) for x, y in slice_points]

        slice_points_fg = [point for point in points_foreground if point[0] == current_slice]
        slice_points_fg = [point for _, point in slice_points_fg]
        slice_points_fg = [(int(round(y)), int(round(x))) for x, y in slice_points_fg]
        
        mask = segmentation.laplacian(image[..., current_slice], slice_points_fg, slice_points)
        new_image = image[..., current_slice] * mask
        image[..., current_slice] = new_image
        clear_points()
        # ax.cla()
        # ax.imshow(new_image, cmap='gray')
        # fig.canvas.draw()

    clear_drawing()
    ax.imshow(image[..., current_slice])
    fig.canvas.draw()
    check_draw()


def on_preprocessing_selected(label):
    """Actualizar el algoritmo de preprocesamiento seleccionado."""
    selected_preprocessing[0] = label
    print(f"Algoritmo de preprocesamiento seleccionado: {label}")

def apply_preprocessing():
    global image
    if image_path is None:
        print("No hay imagen cargada.")
        return

    algorithm = selected_preprocessing[0]
    print(f"Aplicando preprocesamiento: {algorithm}...")

    if algorithm == "Mean filter":
        image = preprocessing.mean_filter(image, 2)
    elif algorithm == "Normalization":
        image = preprocessing.normalize(image)
    elif algorithm == "Bias Fields":
        image = preprocessing.bias_field_correction(image, 50)
    elif algorithm == "N3":
        image = preprocessing.n3(image, 10)
    elif algorithm == "Isotropic Diffusion":
        image = preprocessing.isotropic_diffusion(image)

    clear_drawing()
    ax.imshow(image[..., current_slice])
    fig.canvas.draw()
    check_draw()

radio_buttons.on_clicked(on_algorithm_selected)
preprocessing_radio_buttons.on_clicked(on_preprocessing_selected)
preprocessing_btn.on_clicked(lambda event: apply_preprocessing())
apply_btn.on_clicked(apply_segmentation)
load_btn.on_clicked(open_file)
load_zip_btn.on_clicked(load_zip_event)
save_btn.on_clicked(save_zip_event)

fig.canvas.mpl_connect('scroll_event', update_image)
fig.canvas.mpl_connect('button_press_event', draw_or_erase)
fig.canvas.mpl_connect('motion_notify_event', draw_continuous)

plt.show()
