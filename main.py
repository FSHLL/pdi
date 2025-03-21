import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter
from tkinter import filedialog
import json
import zipfile
import os

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


def load_nifti(file_path):
    global image, image_path
    image_path = file_path
    image_data = nib.load(file_path)
    image = image_data.get_fdata()
    ax.clear()
    fig.canvas.draw_idle()
    ax.imshow(image[..., current_slice])
    fig.canvas.draw()

def open_file(event):
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii *.nii.gz")])
    if file_path:
        load_nifti(file_path)

def update_image(event):
    global current_slice
    current_slice += 1 if event.button == 'up' else -1
    if current_slice >= image.shape[2]:
        current_slice = 0
    clear_drawing()
    ax.imshow(image[..., current_slice])
    fig.canvas.draw()
    check_draw()

def start_drawing(event):
    if event.inaxes == ax:
        draw_point(event)

def draw_continuous(event):
    if event.button and event.inaxes == ax:
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

def check_draw():
    slice_points = [point for point in points if point[0] == current_slice]
    for point in slice_points:
        x, y = point[1]
        ax.plot(x, y, 'ro', markersize=5)
    fig.canvas.draw()

def save_zip(event):
    if image_path is None or not points:
        print("No hay datos para guardar.")
        return

    zip_path = filedialog.asksaveasfilename(
        defaultextension=".zip",
        filetypes=[("ZIP files", "*.zip")],
        title="Guardar ZIP"
    )

    if not zip_path:
        return

    json_data = {"points": points}
    json_path = "puntos.json"
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(image_path, os.path.basename(image_path))
        zipf.write(json_path, json_path)

    os.remove(json_path)

    print(f"Archivo guardado en {zip_path}")

def load_zip(event):
    global image_path, points
    print('INNNNNNNN')
    zip_path = filedialog.askopenfilename(
        filetypes=[("ZIP files", "*.zip")],
        title="Seleccionar archivo ZIP"
    )

    if not zip_path:
        return

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        extract_dir = "temp_extract"
        zipf.extractall(extract_dir)

        nifti_file = None
        json_file = None

        for filename in os.listdir(extract_dir):
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                nifti_file = os.path.join(extract_dir, filename)
            elif filename == "puntos.json":
                json_file = os.path.join(extract_dir, filename)

        if nifti_file:
            load_nifti(nifti_file)

        if json_file:
            with open(json_file, "r") as json_data:
                data = json.load(json_data)
                points = data.get("points", [])

        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
        os.rmdir(extract_dir)

        print(f"Archivo ZIP {zip_path} cargado correctamente")

fig.canvas.mpl_connect('scroll_event', update_image)
fig.canvas.mpl_connect('button_press_event', start_drawing)
fig.canvas.mpl_connect('motion_notify_event', draw_continuous)

load_btn.on_clicked(open_file)
load_zip_btn.on_clicked(load_zip)
save_btn.on_clicked(save_zip)

plt.show()
