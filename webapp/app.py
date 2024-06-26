from flask import Flask, render_template, request, redirect, url_for,session,jsonify,send_from_directory
import os
from werkzeug.utils import secure_filename 
import skimage
import pickle
import base64 
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import tensorflow as tf
# Configuraciones para el entrenamiento y modelo 
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib 
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import time 
from flask_socketio import SocketIO, send
from keras.callbacks import Callback
from os import listdir
import threading
from entrenamiento import train_model
from multiprocessing import Process, Queue
from threading import Thread
import subprocess
import logging
import re


tf.config.set_visible_devices([], 'GPU')
class IgnoreCUPTILogFilter(logging.Filter):
    def filter(self, record):
        # Aquí defines las condiciones para ignorar el mensaje
        return "cupti_interface_->" not in record.msg and "Invalid GPU compute capability" not in record.msg

# Configuración básica de TensorFlow para reducir la cantidad de mensajes de log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger('tensorflow')
logger.addFilter(IgnoreCUPTILogFilter())

process_status = {}




app = Flask(__name__)
app.config['MODEL_UPLOAD_FOLDER'] = 'weights' 
socketio = SocketIO(app, cors_allowed_origins="*")
app.secret_key = 'una_clave_secreta_very_secret'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'h5'}


def create_or_get_session_id():
    if 'session_id' not in session:
        # Genera un identificador único para la sesión
        session['session_id'] = os.urandom(16).hex()
    return session['session_id']

def update_status(session_id, status, progress):
    if session_id not in process_status:
        process_status[session_id] = {}
    process_status[session_id]['status'] = status
    process_status[session_id]['progress'] = int(progress)


@app.route('/get_status')
def get_status():
    session_id = create_or_get_session_id()
    status = process_status.get(session_id, {'status': 'Listo', 'progress': 0})
    return jsonify(status)


class PredictionConfig(Config):
	# definimos nombre de configurcion
	NAME = "insects_cfg"
	# tendremos 3 clases (3 insectos) + 1 que seria el fondo
	NUM_CLASSES = 1 + 3
	# Especificamos que utilziaremos una sola GPU 
	GPU_COUNT = 1
    # Indicamos cuantas imagenes se procesaran por GPU a la vez
	IMAGES_PER_GPU = 1


# preparemos la configuracion
cfg = PredictionConfig()
cfg.display() # aseguramos de la correcta configuracion

# prepareamos el directorio donde se guardara el modelo
import os
ROOT_DIR = os.path.abspath("./")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "weights")
 
##############

# Iniciamos el modelo con la configuracion previa 



model = MaskRCNN(mode='inference', model_dir='weights', config=cfg, optimizer='SGD')
#model.load_weights('weights/mask_rcnn_insects_cfg_24_04_2024.h5', by_name=True)
# Ruta a la página principal
@app.route('/')
def index():
    return render_template('index.html')

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


@app.route('/deteccion', methods=['GET', 'POST'])
def deteccion():
    pngImageB64String = ''
    conteos = []  # Inicialización de conteos

    if request.method == 'POST':
        try:
            session_id = create_or_get_session_id()
            print("Solicitud recibida")
            file = request.files.get('fileInput')
            
            if not file or file.filename == '':
                print("No se seleccionó ningún archivo")
                return 'No se seleccionó ningún archivo', 400

            modeloCargadoUsuario = request.files.get('modelInput')
            pesos = request.form.get('model-option')
            
            if modeloCargadoUsuario and allowed_file(modeloCargadoUsuario.filename):
                filename = secure_filename(modeloCargadoUsuario.filename)
                model_path = os.path.join(app.config['MODEL_UPLOAD_FOLDER'], filename)
                modeloCargadoUsuario.save(model_path)
                print(f"Modelo guardado en {model_path}")

                if pesos == "custom":
                    print("Cargando modelo personalizado")
                    model.load_weights(model_path, by_name=True)
            elif pesos == "default":
                print("Cargando modelo por defecto")
                model.load_weights('weights/mask_rcnn_insects_cfg_24_04_2024.h5', by_name=True)
            else:
                return "Opción de modelo no válida o archivo faltante", 400

            print("cargando imagen...")
            insect_img = skimage.io.imread(file.stream)

            print("detectando...")
            detected = model.detect([insect_img])[0]

            print("graficando resultados...")
            fig, ax = pyplot.subplots()
            display_instances(insect_img, detected['rois'], detected['masks'], detected['class_ids'], ['BG', 'WF', 'NC', 'MR'], detected['scores'], ax=ax)

            pngImage = BytesIO()
            FigureCanvas(fig).print_png(pngImage)
            pyplot.close(fig)

            print("codificando...")
            pngImageB64String = "data:image/png;base64," + base64.b64encode(pngImage.getvalue()).decode('utf8')

            conteos = count_class_ids(detected['class_ids'])
            update_status(session_id, 'Proceso completado con éxito.', 100)

            return render_template('deteccion.html', image_data=pngImageB64String, conteos=conteos)
        except Exception as e:
            print("Error durante la detección o la visualización:", e)
            return str(e), 500

    return render_template('deteccion.html', image_data=pngImageB64String, conteos=conteos)



#####################################################################################################################

def start_training_thread(filename,algorithm,epoch,steps,proporcionEntrenamiento,proporcionTest,Resnet,validacion,):
    # Define el patrón de las expresiones regulares para capturar las líneas relevantes
    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
    batch_pattern = re.compile(r"(\d+)/(\d+)\s+\[.*?\]\s+-\s+.*?\s+-\s+loss:\s+([\d\.]+)(?:\s+-\s+val_loss:\s+([\d\.]+))?")

    # Inicia el proceso de entrenamiento

    data = {
        'filename': filename,
        'algorithm': algorithm,
        'epoch': epoch,
        'steps': steps,
        'proporcionEntrenamiento': proporcionEntrenamiento,
        'proporcionTest': proporcionTest,
        'Resnet': Resnet,
        'validacion': validacion
    }
    send_data_to_frontend(data)

    
    process = subprocess.Popen(['python', 'entrenamiento.py', filename,algorithm,epoch,steps,proporcionEntrenamiento,proporcionTest,Resnet,validacion], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8')

    while True:
        send_data_to_frontend(data)
        output = process.stdout.readline()
        print("Debug: Output del proceso", output)
        if output == '' and process.poll() is not None:
            break
        if "Model training completed and saved." in output:
            socketio.emit('training_complete', {'file': filename +".h5"}, namespace='/entrenamiento')
            print("Enviando al front - Entrenamiento completado")
        if output:
            epoch_match = epoch_pattern.search(output)
            batch_match = batch_pattern.search(output)
            if epoch_match:
                epoch_data = {'current_epoch': epoch_match.group(1), 'total_epochs': epoch_match.group(2)}
                socketio.emit('update_epoch', epoch_data, namespace='/entrenamiento')
            elif batch_match:
                batch_data = {'current_batch': batch_match.group(1), 'total_batches': batch_match.group(2), 'loss': batch_match.group(3)}
                socketio.emit('update_batch', batch_data, namespace='/entrenamiento')

    process.stdout.close()
    process.stderr.close()
    process.wait()


@app.route('/download/<filename>')
def download_file(filename):
    # Asegúrate de validar o sanitizar el nombre del archivo para seguridad
    directory = os.path.join(os.getcwd(), 'weights')
    return send_from_directory(directory, filename, as_attachment=True)


def send_data_to_frontend(data):
    socketio.emit('update_data', data, namespace='/entrenamiento')

@app.route('/entrenamiento', methods=['GET', 'POST'])
def entrenamiento():
    if request.method == 'POST':
        filename = request.form['Name']
        algorithm = request.form['algoritmo']
        epoch = request.form['epoch']
        steps = request.form['steps']
        proporcionEntrenamiento = (request.form['proporcionesValue'])
        proporcionTest = (request.form['testValue'])
        Resnet = (request.form['modelo_resnet'])
        validacion = request.form['validacion']

        data = {
            'filename': filename,
            'algorithm': algorithm,
            'epoch': epoch,
            'steps': steps,
            'proporcionEntrenamiento': proporcionEntrenamiento,
            'proporcionTest': proporcionTest,
            'Resnet': Resnet,
            'validacion': validacion
        }
        send_data_to_frontend(data)

        
        print("Filename:", filename)
        print("Algorithm:", algorithm)
        print("Training Proportion:", proporcionEntrenamiento)
        print("Test Proportion:", proporcionTest)
        print("el modelo del resnet llegado al backend es" ,Resnet )
        print("la validacion que llega es" , validacion)
        
        training_thread = Thread(target=start_training_thread, args=(filename,algorithm,epoch,steps,proporcionEntrenamiento,proporcionTest,Resnet,validacion,))
        training_thread.start()
    return render_template('entrenamiento.html')



@socketio.on('connect', namespace='/entrenamiento')
def test_connect():
    print("Cliente conectado.")
    send({'data': 'Connected'}, namespace='/entrenamiento', json=True)  # Use `send` with json=True

@app.route('/test')
def test():
    socketio.send({'message': 'Hello, this is a test!'}, namespace='/entrenamiento', json=True)  # Use `send` with json=True
    return "Message sent!"

@app.route('/info')
def info():
    """
    Ruta para mostrar la página de información.
    """
    return render_template('info.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)    