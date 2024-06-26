import unittest
from mrcnn.config import Config
from insect_mask_rcnn import InsectsConfig

class TestConfig(unittest.TestCase):

    def test_config_attributes(self):
        # Prueba que los atributos de la configuración están configurados correctamente
        config = InsectsConfig()
        self.assertEqual(config.NAME, "insects_cfg", "El nombre de la configuración es incorrecto")
        self.assertEqual(config.NUM_CLASSES, 4, "El número de clases es incorrecto")
        self.assertEqual(config.STEPS_PER_EPOCH, 5, "Los pasos por época son incorrectos")

if __name__ == '__main__':
    unittest.main()