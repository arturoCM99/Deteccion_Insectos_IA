import unittest
from insect_mask_rcnn import InsectsDataset
import numpy as np

class TestLoadMask(unittest.TestCase):

    def setUp(self):
        # Configura el entorno antes de cada prueba
        self.dataset = InsectsDataset()
        self.dataset_dir = 'datasets/renamed_to_numbers'
        self.dataset.load_dataset(self.dataset_dir, is_train=True)
        self.dataset.prepare()

    def test_load_mask(self):
        # Prueba que las máscaras se cargan correctamente
        image_id = self.dataset.image_ids[0]
        mask, class_ids = self.dataset.load_mask(image_id)
        self.assertIsInstance(mask, np.ndarray, "La máscara no es un ndarray")
        self.assertIsInstance(class_ids, np.ndarray, "Los IDs de las clases no son un ndarray")
        self.assertEqual(mask.shape[-1], len(class_ids), "La cantidad de máscaras y IDs de clase no coincide")

if __name__ == '__main__':
    unittest.main()
