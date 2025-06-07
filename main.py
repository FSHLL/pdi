import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons
import tkinter
import segmentation
import preprocessing
import register
from tkinter import filedialog
from tools import load_nifti, save_zip, load_zip

window = tkinter.Tk()
window.withdraw()

panel_left = 0.05
button_width = 0.2
button_height = 0.06
spacing = 0.01

y_pos = 0.9

image_path = None
current_slice = 0
points = []
points_foreground = []
region_growing_enabled = False

second_image = None
second_image_path = None
current_slice_second = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)

# Cargar imagen
ax_load_btn = plt.axes([panel_left, y_pos, button_width, button_height])
load_btn = Button(ax_load_btn, 'Cargar NIfTI')

y_pos -= (button_height + spacing)
ax_load_zip_btn = plt.axes([panel_left, y_pos, button_width, button_height])
load_zip_btn = Button(ax_load_zip_btn, 'Cargar ZIP')

y_pos -= (button_height + spacing)
ax_save_btn = plt.axes([panel_left, y_pos, button_width, button_height])
save_btn = Button(ax_save_btn, 'Guardar ZIP')

y_pos -= (button_height + spacing * 2)
ax_load_second_btn = plt.axes([panel_left, y_pos, button_width, button_height])
load_second_btn = Button(ax_load_second_btn, 'Cargar NIfTI 2')

y_pos -= (button_height + spacing)
ax_register_btn = plt.axes([panel_left, y_pos, button_width, button_height])
register_btn = Button(ax_register_btn, 'Registrar')

segmentation_algorithms = ['Thresholding', 'Isodata', 'Region Growing', 'K-means', 'Laplacian']
selected_algorithm = [segmentation_algorithms[0]]

y_pos -= (button_height + spacing)
ax_radio = plt.axes([panel_left, y_pos - 0.1, button_width, 0.1], frameon=True)
radio_buttons = RadioButtons(ax_radio, segmentation_algorithms)

y_pos -= (0.1 + spacing)
ax_apply_btn = plt.axes([panel_left, y_pos - button_height, button_width, button_height])
apply_btn = Button(ax_apply_btn, 'Aplicar Segmentación')

preprocessing_algorithms = ['Mean filter', 'Median filter', 'Edge preservation', 'Non-local mean']
selected_preprocessing = [preprocessing_algorithms[0]]

y_pos -= (button_height + spacing * 2)
ax_preprocessing_radio = plt.axes([panel_left, y_pos - 0.2, button_width, 0.2], frameon=True)
preprocessing_radio_buttons = RadioButtons(ax_preprocessing_radio, preprocessing_algorithms)

y_pos -= (0.2 + spacing)
ax_preprocessing_btn = plt.axes([panel_left, y_pos - button_height, button_width, button_height])
preprocessing_btn = Button(ax_preprocessing_btn, 'Preprocesar')

def open_file(event):
    global image, image_path, current_slice
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    if file_path:
        image_path = file_path
        image = load_nifti(file_path, current_slice, ax1, fig)
        current_slice = image.shape[2] // 2
        clear_drawing()
        ax1.imshow(image[..., current_slice])
        fig.canvas.draw()
        check_draw()
        ax1.set_title("Imagen Original")

def open_second_file(event):
    global second_image, second_image_path, current_slice_second
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    if file_path:
        second_image_path = file_path
        second_image = load_nifti(file_path, current_slice_second, ax2, fig)
        current_slice_second = second_image.shape[2] // 2

        ax2.cla()
        ax2.imshow(second_image[..., current_slice_second])
        fig.canvas.draw()
        ax2.set_title("Imagen a Registrar")

def apply_register(event):
    global second_image
    second_image = register.apply_registration(image, second_image)
    current_slice_second = second_image.shape[2] // 2
    ax2.cla()
    ax2.imshow(second_image[..., current_slice_second])
    ax2.set_title("Imagen a Registrar")
    fig.canvas.draw()

def update_image(event):
    global current_slice, current_slice_second
    
    direction = 1 if event.button == 'up' else -1

    if event.inaxes == ax1:
        if image is None:
            return
        current_slice = (current_slice + direction)
        clear_drawing()
        ax1.imshow(image[..., current_slice], cmap='gray')
        ax1.set_title("Imagen Original")
        check_draw()
    
    elif event.inaxes == ax2:
        if second_image is None:
            return
        current_slice_second = (current_slice_second + direction)
        ax2.cla()
        ax2.imshow(second_image[..., current_slice_second], cmap='gray')
        ax2.set_title("Imagen a Registrar")
        fig.canvas.draw()

def draw_or_erase(event):
    if event.inaxes == ax1 and event.button == 1:
        draw_point(event)
    elif event.inaxes == ax1 and event.button == 3:
        draw_point_foreground(event)

def draw_continuous(event):
    if event.inaxes == ax1 and event.button == 1:
        draw_point(event)

def clear_drawing():
    ax1.cla()
    ax1.imshow(image[..., current_slice])
    fig.canvas.draw()

def draw_point(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        points.append((current_slice, (x, y)))
        ax1.plot(x, y, 'ro', markersize=5)
        fig.canvas.draw()

def draw_point_foreground(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        points_foreground.append((current_slice, (x, y)))
        ax1.plot(x, y, 'ro', markersize=5, color='blue')
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
        ax1.plot(x, y, 'ro', markersize=5)
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
            image = load_nifti(nifti_file, current_slice, ax1, fig)
            current_slice = image.shape[2] // 2
            clear_drawing()
            ax1.imshow(image[..., current_slice])
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

    if algorithm == "Thresholding":
        image = segmentation.simple_thresholding(image, 0.00001)
    elif algorithm == "Isodata":
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
    ax1.imshow(image[..., current_slice])
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
        image = preprocessing.mean_filter(image, 3)
    elif algorithm == "Median filter":
        image = preprocessing.median_filter(image, 3)
    elif algorithm == "Edge preservation":
        image = preprocessing.anisotropic_diffusion(image, num_iter=10, kappa=30, gamma=0.1)
    elif algorithm == "Non-local mean":
        image = preprocessing.non_local_means_filter(image, patch_size=1, search_size=3, h=0.1)
    elif algorithm == "Isotropic Diffusion":
        image = preprocessing.isotropic_diffusion(image)

    clear_drawing()
    ax1.imshow(image[..., current_slice])
    fig.canvas.draw()
    check_draw()

radio_buttons.on_clicked(on_algorithm_selected)
preprocessing_radio_buttons.on_clicked(on_preprocessing_selected)
preprocessing_btn.on_clicked(lambda event: apply_preprocessing())
apply_btn.on_clicked(apply_segmentation)
load_btn.on_clicked(open_file)
load_zip_btn.on_clicked(load_zip_event)
save_btn.on_clicked(save_zip_event)

load_second_btn.on_clicked(open_second_file)
register_btn.on_clicked(apply_register)

fig.canvas.mpl_connect('scroll_event', update_image)
fig.canvas.mpl_connect('button_press_event', draw_or_erase)
fig.canvas.mpl_connect('motion_notify_event', draw_continuous)

plt.show()
