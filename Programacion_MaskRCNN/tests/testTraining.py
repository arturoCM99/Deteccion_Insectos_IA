import unittest
from insect_mask_rcnn import InsectsConfig, InsectsDataset
from mrcnn.model import MaskRCNN
import os

class TestTraining(unittest.TestCase):

    def setUp(self):
        # Configura el entorno antes de cada prueba
        self.config = InsectsConfig()
        self.train_set = InsectsDataset()
        self.train_set.load_dataset('datasets/renamed_to_numbers', is_train=True)
        self.train_set.prepare()
        self.test_set = InsectsDataset()
        self.test_set.load_dataset('datasets/renamed_to_numbers', is_train=False)
        self.test_set.prepare()

    def test_train_model(self):
        # Prueba el entrenamiento del modelo
        model = MaskRCNN(mode='training', model_dir="logs", config=self.config, optimizer='SGD')
        model.load_weights("weights/mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
        model.train(self.train_set, self.test_set, learning_rate=self.config.LEARNING_RATE, epochs=1, layers='heads')
        
        model_path = './logs/mask_rcnn_insects_cfg_test.h5'
        model.keras_model.save_weights(model_path)
        self.assertTrue(os.path.exists(model_path), "No se guardaron los pesos del modelo")

if __name__ == '__main__':
    unittest.main()
