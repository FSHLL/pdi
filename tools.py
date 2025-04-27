import nibabel as nib
import json
import zipfile
import os

def load_nifti(file_path, current_slice, ax, fig):
    image_data = nib.load(file_path)
    image = image_data.get_fdata()
    ax.clear()
    fig.canvas.draw_idle()
    ax.imshow(image[..., current_slice])
    fig.canvas.draw()
    return image

def save_zip(image_path, points, zip_path):
    if image_path is None or not points:
        raise ValueError("No hay datos para guardar.")

    json_data = {"points": points}
    json_path = "puntos.json"
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(image_path, os.path.basename(image_path))
        zipf.write(json_path, json_path)

    os.remove(json_path)

def load_zip(zip_path):
    extract_dir = "temp_extract"
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_dir)

        nifti_file = None
        json_file = None

        for filename in os.listdir(extract_dir):
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                nifti_file = os.path.join(extract_dir, filename)
            elif filename == "puntos.json":
                json_file = os.path.join(extract_dir, filename)

        points = []
        if json_file:
            with open(json_file, "r") as json_data:
                data = json.load(json_data)
                points = data.get("points", [])

        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
        os.rmdir(extract_dir)

        return nifti_file, points