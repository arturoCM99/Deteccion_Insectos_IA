import unittest
from insect_mask_rcnn import InsectsDataset
import os

class TestInsectsDataset(unittest.TestCase):

    def setUp(self):
        # Configura el entorno antes de cada prueba
        self.dataset = InsectsDataset()
        self.dataset_dir = os.path.join('datasets', 'renamed_to_numbers')

    def test_load_images(self):
        # Prueba que todas las imágenes se cargan correctamente
        self.dataset.load_dataset(self.dataset_dir, is_train=True)
        self.dataset.prepare()
        self.assertGreater(len(self.dataset.image_ids), 0, "No se cargaron imágenes")

    def test_load_annotations(self):
        # Prueba que las anotaciones se cargan correctamente
        self.dataset.load_dataset(self.dataset_dir, is_train=True)
        self.dataset.prepare()
        for image_id in self.dataset.image_ids:
            annotation = self.dataset.image_info[image_id]['annotation']
            self.assertIsNotNone(annotation, "Falta la anotación para la imagen {}".format(image_id))

if __name__ == '__main__':
    unittest.main()
