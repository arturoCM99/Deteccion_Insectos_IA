import os
from xml.etree import ElementTree
from PIL import Image

# Define el directorio donde están almacenadas tus imágenes y archivos XML
# Reemplaza con la ruta completa si el script no se ejecuta en la misma ubicación
directorio_dataset = 'C:/Users/Arturo/Desktop/TFG/datasets/renamed_to_numbers'
directorio_imagenes = os.path.join(directorio_dataset, 'images')
directorio_anotaciones = os.path.join(directorio_dataset, 'annots')

# Lista todos los archivos jpg en el directorio de imágenes
archivos_imagen = [f for f in os.listdir(directorio_imagenes) if f.endswith('.jpg')]

# Esta función actualiza las dimensiones en las anotaciones XML
def actualizar_dimensiones_anotacion(archivo_xml, ancho_img, alto_img):
    # Analiza el archivo XML
    arbol = ElementTree.parse(archivo_xml)
    raiz = arbol.getroot()
    
    # Actualiza el elemento de tamaño
    elemento_tamano = raiz.find('size')
    elemento_ancho = elemento_tamano.find('width')
    elemento_alto = elemento_tamano.find('height')
    elemento_ancho.text = str(ancho_img)
    elemento_alto.text = str(alto_img)
    
    # Guarda los cambios de vuelta en el archivo
    arbol.write(archivo_xml)

# Procesa cada archivo de imagen
for archivo_img in archivos_imagen:
    # Construye la ruta completa para el archivo de imagen y cárgalo
    ruta_img = os.path.join(directorio_imagenes, archivo_img)
    with Image.open(ruta_img) as img:
        ancho_img, alto_img = img.size
        
    # Construye la ruta correspondiente del archivo XML basado en el nombre del archivo de imagen
    nombre_archivo_xml = os.path.splitext(archivo_img)[0] + '.xml'
    ruta_archivo_xml = os.path.join(directorio_anotaciones, nombre_archivo_xml)
    
    # Actualiza las dimensiones de la anotación
    if os.path.exists(ruta_archivo_xml):
        actualizar_dimensiones_anotacion(ruta_archivo_xml, ancho_img, alto_img)
    else:
        print(f"No existe el archivo de anotación {nombre_archivo_xml}.")
