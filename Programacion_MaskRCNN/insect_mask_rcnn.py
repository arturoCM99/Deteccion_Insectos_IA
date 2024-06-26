

import tensorflow as tf

#Para trabajar con CPU 
tf.config.set_visible_devices([], 'GPU')

''' Configuracion asignacion de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        mem_limit = 4096 * 0.9  # 90% de 4096 MB
        
        # Configuramos el límite de memoria virtual para la GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],  # Asumimos que queremos configurar solo la primera GPU encontrada
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(mem_limit))]
        )
    except RuntimeError as e:
        # Esta excepción puede lanzarse si la configuración se intenta después de que la GPU ha sido inicializada
        print(e)
'''  

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

# Configuraciones para el entrenamiento y modelo 
from mrcnn.config import Config
from mrcnn.model import MaskRCNN





    
# herada de Dataset de MaskRCNN 
class InsectsDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # definimos 3 clases de insectos ya que tendremos 3 insectos ha detectar
        self.add_class("dataset", 1, "WF")
        self.add_class("dataset", 2, "NC")
        self.add_class("dataset", 3, "MR")
        
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
       
             
		# find all images
        for filename in listdir(images_dir):
            print(filename)
			# extract image id
            image_id = filename[:-4]
			#print('IMAGE ID: ',image_id)
			
			# skip all images after 196 if we are building the train set
            if is_train and int(image_id) >= 196: # 70%
                continue
			# skip all images before 196 if we are building the val set
            if not is_train and int(image_id) < 196: # 30%
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [0,1,2,3])


	# extrae los bordes de las cajas delimitadoras 
    def extract_boxes(self, filename):
		# load and parse the file
        tree = ElementTree.parse(filename)
		# get the root of the document
        root = tree.getroot()
		# extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text   #Add label name to the box list
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
		# extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

	# generamos las mascaras para cada uno de los insectos 
    def load_mask(self, image_id):
		# obtenemos informacion de la imagen como id_imagen ,dimensiones , anotacion correspondiente , ids_de los insectos presentes...
        info = self.image_info[image_id]
		# definimos las ubicaciones de las anotaciones 
        path = info['annotation']
        #return info, path
        
        
		# cargamos el .XML
        boxes, w, h = self.extract_boxes(path) #extraemos el ancho y alto de la caja delimitadora
		# creamos un array de 0 como mascara 
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # guardamos los id de los insectos
        class_ids = list()
        # Repetimso proceso para cada una de las  cajas 
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            
            
            # El parametro quinto  recibira el nombre de la caja en este caso de su insecto correspondiente
            if (box[4] == 'WF'): # mosca blanca
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('WF'))
            elif(box[4] == 'NC'): # nesidicoris
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('NC')) 
            elif(box[4] == 'MR'): # macrolophus
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('MR'))
          
        return masks, asarray(class_ids, dtype='int32')
        

	# cargamos la imagen de referencia
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# train conjunto
dataset_dir='datasets/renamed_to_numbers'

train_set = InsectsDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val conjunto
test_set = InsectsDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


import random
num=random.randint(0, len(train_set.image_ids))
# define image id
image_id = num
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)



# definimos configuracion del modelo. hereda de config libreria de mask rcnn
class InsectsConfig(Config):
	# definimos nombre de la configuracion
	NAME = "insects_cfg"
	# tendremos 3 clases (3 insectos) + 1 que seria el fondo
	NUM_CLASSES = 1 + 3
	# Steps(pasos) numero de imagenes para que el modelo actualice sus pesos 
    # Calcula el gradiante de la funcion de perdida respecto a los pesos del modelo y los actualiza segun el algoritmo
    # El algoritmo por defecto es Stochastic Gradient Descent (SGD) o Descenso del Gradiente Estocástico.
    # Epoca pasada completa del conjunto de datos
	STEPS_PER_EPOCH = 100
    # Tasa de aprendizaje nos permite establecer CUANTO se actulizan los pesos
    # Posibles problemas: 
    # Si es bajo, aprendizaje lento y minimo locales
    # Si es alto , entrenamiento inestable
    # learning_rate = 0.001 # por defecto hereda del config
    

# preparemos la configuracion
config = InsectsConfig()
config.display() # aseguramos de la correcta configuracion

# prepareamos el directorio donde se guardara el modelo
import os
ROOT_DIR = os.path.abspath("./")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")




###############
optimizador = "Adam" #SGD , Adam , RMSprop
# Iniciamos el modelo con la configuracion previa 
model = MaskRCNN(mode='training', model_dir="logs", config=config ,optimizer=optimizador)
# cargamos los pesos.Por defecto partimos de unos iniciales y no de 0 
#  COCO, que significa "Common Objects in Context" de Microsoft 
model.load_weights("weights/mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])




# entrenamiento del  modelo
# epoca define total de pasadas del conjunto de datos 
# layers indica que capas del modelo han de ser entrenadas
# test_set corresponde con nuestro conjunto de datos definidos previamente
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')
# guardamos en el directorio logs los pesos del entrenamiento
model_path = './weights/mask_rcnn_insects_cfg_24_04_2025.h5'
model.keras_model.save_weights(model_path)


training_losses = model.losses_per_step
validation_losses = model.val_losses_per_epoch

# Imprimir pérdidas
print("Training Losses per Step:", training_losses)
print("Validation Losses per Epoch:", validation_losses)

# Graficar las pérdidas
model.plot_losses()

#########################################


from matplotlib.patches import Rectangle # para dibujar los rectangulos delimitadores de los insectos detectados

#Hereda de mrcnn config
class PredictionConfig(Config):
	# definimos nombre de configurcion
	NAME = "insects_cfg"
	# tendremos 3 clases (3 insectos) + 1 que seria el fondo
	NUM_CLASSES = 1 + 3
	# Especificamos que utilziaremos una sola GPU 
	GPU_COUNT = 1
    # Indicamos cuantas imagenes se procesaran por GPU a la vez
	IMAGES_PER_GPU = 1
 


# creamos la configuracion del modelo
cfg = PredictionConfig()
# definimos el modelo 
# Para utilizar SGD es el que utiliza Mask RCNN por defecto
import tensorflow as tf

model = MaskRCNN(mode='inference', model_dir='logs', config=cfg, optimizer='SGD')


model.load_weights('weights/mask_rcnn_insects_cfg_24_04_2024.h5', by_name=True)


import skimage

# cojemos la imagen de entrada en la que se intetara predecir que insectos hay
insect_img = skimage.io.imread("datasets/renamed_to_numbers/images/001.jpg") 

#insect_img = skimage.io.imread("datasets/284.jpg")
detected = model.detect([insect_img])[0] 

# Mostramos los resultados de los nisectos identificados
pyplot.imshow(insect_img)
ax = pyplot.gca()
class_names = ['BG', 'WF', 'NC', 'MR']  # Agregamos 'BG' al inicio para el fondo
display_instances(insect_img, detected['rois'],detected['masks'],detected['class_ids'],class_names,detected['scores'])

def count_class_ids(class_ids):
    class_id_counter_WF = 0  
    class_id_counter_NC = 0  
    class_id_counter_MR = 0  
    class_id_counter_TOTAL = 0  # Contador para el total de elementos

    # Iteramos sobre cada elemento en el array para contar las ocurrencias
    for class_id in class_ids:
        if class_id == 1:
            class_id_counter_WF += 1
        elif class_id == 2:
            class_id_counter_NC += 1
        elif class_id == 3:
            class_id_counter_MR += 1
        class_id_counter_TOTAL += 1  # Incrementamos el contador total en cada iteración

    # Retornamos una lista con los contadores
    return [class_id_counter_WF, class_id_counter_NC, class_id_counter_MR, class_id_counter_TOTAL]


# Ejemplo de uso:
class_ids = [2, 2, 2, 2, 2, 2, 3, 3, 2]
resultados = count_class_ids(class_ids)
print(resultados) 




print(detected['class_ids'])
class_id_counter=1
for box in detected['rois']:
    detected_class_id = detected['class_ids'][class_id_counter-1]
    y1, x1, y2, x2 = box
    #calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    #create the shape
    ax.annotate(class_names[detected_class_id-1], (x1, y1), color='black', weight='bold', fontsize=10, ha='center', va='center')
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#draw the box
    ax.add_patch(rect)
    class_id_counter+=1