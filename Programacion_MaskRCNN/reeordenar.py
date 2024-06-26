import os
import glob
import random

class DatasetRandomizer:
    def randomize_dataset(self, images_dir, annot_dir, seed=42):
        # Obtener lista de archivos de imágenes
        imagepath_list = glob.glob(os.path.join(images_dir, '*.jpg'))
        if not imagepath_list:
            print(f"No images found in directory: {images_dir}")
            return

        print(f"Found {len(imagepath_list)} image files.")
        
        random.Random(seed).shuffle(imagepath_list)  # Set the seed for reproducibility

        padding = len(str(len(imagepath_list)))  # Number of digits to add for file number
        print(f"Padding length for filenames: {padding}")

        # Renombrar a nombres temporales
        for n, filepath in enumerate(imagepath_list, 1):
            temp_filepath = os.path.join(images_dir, f'temp_{n:0{padding}}.jpg')
            os.rename(filepath, temp_filepath)
            print(f"Temporarily renamed image: {filepath} to {temp_filepath}")  # Mensaje de verificación

        # Renombrar a nombres finales
        temp_imagepath_list = glob.glob(os.path.join(images_dir, 'temp_*.jpg'))
        for n, temp_filepath in enumerate(temp_imagepath_list, 1):
            new_filepath = os.path.join(images_dir, f'{n:0{padding}}.jpg')
            os.rename(temp_filepath, new_filepath)
            print(f"Renamed image: {temp_filepath} to {new_filepath}")  # Mensaje de verificación

        # Obtener lista de archivos de anotaciones
        annotpath_list = glob.glob(os.path.join(annot_dir, '*.xml'))
        if not annotpath_list:
            print(f"No annotations found in directory: {annot_dir}")
            return

        print(f"Found {len(annotpath_list)} annotation files.")

        random.Random(seed).shuffle(annotpath_list)

        # Renombrar a nombres temporales
        for m, filepath in enumerate(annotpath_list, 1):
            temp_filepath = os.path.join(annot_dir, f'temp_{m:0{padding}}.xml')
            os.rename(filepath, temp_filepath)
            print(f"Temporarily renamed annotation: {filepath} to {temp_filepath}")  # Mensaje de verificación

        # Renombrar a nombres finales
        temp_annotpath_list = glob.glob(os.path.join(annot_dir, 'temp_*.xml'))
        for m, temp_filepath in enumerate(temp_annotpath_list, 1):
            new_filepath = os.path.join(annot_dir, f'{m:0{padding}}.xml')
            os.rename(temp_filepath, new_filepath)
            print(f"Renamed annotation: {temp_filepath} to {new_filepath}")  # Mensaje de verificación

# Directorios de imágenes y anotaciones
images_dir = 'datasets/renamed_to_numbers/images'
annot_dir = 'datasets/renamed_to_numbers/annots'

# Crear instancia de la clase
randomizer = DatasetRandomizer()

# Llamar al método randomize_dataset
randomizer.randomize_dataset(images_dir, annot_dir, seed=42)

# Verificación manual
print("\nVerifica manualmente que los archivos han sido renombrados en los directorios correspondientes.")

