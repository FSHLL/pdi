import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter
from tkinter import filedialog
from tools import load_nifti, save_zip, load_zip

window = tkinter.Tk()
window.withdraw()

image_path = None
current_slice = 0
points = []

fig, ax = plt.subplots()

ax_load_btn = plt.axes([0.1, 0.9, 0.2, 0.075])
load_btn = Button(ax_load_btn, 'Cargar NIfTI')

ax_load_zip_btn = plt.axes([0.5, 0.9, 0.2, 0.075])
load_zip_btn = Button(ax_load_zip_btn, 'Cargar ZIP')

ax_save_btn = plt.axes([0.7, 0.9, 0.2, 0.075])
save_btn = Button(ax_save_btn, 'Guardar ZIP')


def open_file(event):
    global image, image_path
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    if file_path:
        image_path = file_path
        image = load_nifti(file_path, current_slice, ax, fig)

def update_image(event):
    global current_slice
    if event.button == 'up':
        current_slice += 1
    elif event.button == 'dow':
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
        erase_point(event)

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
    global image_path, points
    zip_path = filedialog.askopenfilename(
        filetypes=[("ZIP files", "*.zip")],
        title="Seleccionar archivo ZIP"
    )
    if zip_path:
        nifti_file, loaded_points = load_zip(zip_path)
        if nifti_file:
            image_path = nifti_file
            image = load_nifti(nifti_file, current_slice, ax, fig)
        points = loaded_points
        print(f"Archivo ZIP {zip_path} cargado correctamente")

fig.canvas.mpl_connect('scroll_event', update_image)
fig.canvas.mpl_connect('button_press_event', draw_or_erase)
fig.canvas.mpl_connect('motion_notify_event', draw_continuous)

load_btn.on_clicked(open_file)
load_zip_btn.on_clicked(load_zip_event)
save_btn.on_clicked(save_zip_event)

plt.show()
