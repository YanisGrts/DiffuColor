import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

def process_image(filename):
    if filename.lower().endswith('.jpg'):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        try:
            # Ouvrir, redimensionner et sauvegarder
            with Image.open(in_path) as img:
                # LANCZOS offre la meilleure qualité pour la réduction d'image
                img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
                img_resized.save(out_path, format='JPEG', quality=90)
        except Exception as e:
            print(f"Erreur sur l'image {filename}: {e}")

def is_color_image(path, threshold=5.0):
    img = np.array(Image.open(path).convert("LAB"))
    ab_std = img[:, :, 1:].std()  # std of a and b channels
    return

input_dir = 'train2017'        # Le dossier contenant les images originales
output_dir = 'train2017'   # Le dossier de destination

files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

print(f"Début du redimensionnement de {len(files)} images...")

with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(process_image, files), total=len(files)))


output_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]
removed_count = 0

for filename in tqdm(output_files, desc="Filtering B&W images"):
    img_path = os.path.join(output_dir, filename)
    if not is_color_image(img_path):
        os.remove(img_path)
        removed_count += 1



input_dir = 'val2017'        # Le dossier contenant les images originales
output_dir = 'val2017'

files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

print(f"Début du redimensionnement de {len(files)} images...")

with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(process_image, files), total=len(files)))


output_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.jpg')]
removed_count = 0

for filename in tqdm(output_files, desc="Filtering B&W images"):
    img_path = os.path.join(output_dir, filename)
    if not is_color_image(img_path):
        os.remove(img_path)
        removed_count += 1
