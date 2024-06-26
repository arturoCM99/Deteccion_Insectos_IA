# Detección de Insectos en Cultivos

Este proyecto tiene como objetivo la identificación de tres tipos de insectos en cultivos utilizando inteligencia artificial. Estos insectos incluyen la mosca blanca, que es perjudicial para la plantación, así como Nesidiocoris y Macrolophus pygmaeus, ambas especies liberadas con el propósito de controlar la población de mosca blanca en el cultivo.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribución](#contribución)
- [Capturas de Pantalla](#capturas-de-pantalla)
- [Posibles Mejoras](#posibles-mejoras)
- [Licencia](#licencia)


## Descripción

El proyecto utiliza técnicas de aprendizaje profundo para detectar y clasificar insectos en imágenes de cultivos. La red neuronal empleada está basada en la metodología Mask R-CNN.Se programó con la version 3.7.11 de python es posible que si no se utilice con esta versión surgan incomptibilidades entre distintas librerías.

Los modelos preentrenados en el proyecto son de un tamaño muy grande y pueden corromperse durante la subida y descarga para ello se facilita un drive con su descarga en caso de presentarse problemas.
[Enlace a los modelos en Google Drive](https://drive.google.com/drive/folders/1G8NQ3AoleelwGGKPwlqq_IFaKRPRiP7v)

## Instalación

Para instalar el proyecto, se recomienda utilizar un entorno virtual de conda con la versión de Python 3.7.11. Puedes optar por utilizar el script `install.bat` para una instalación automatizada en Windows o seguir los pasos de instalación manual. A continuación, se describen ambos métodos:

### Método 1: Instalación Automatizada en Windows

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tu_usuario/Deteccion_Insectos_TFG.git
    cd Deteccion_Insectos_TFG
    ```

2. Se recomienda crear y activar un entorno virtual con conda como se ha hecho en este caso:
    ```sh
    conda create --name deteccion_insectos python=3.7.11
    conda activate deteccion_insectos
    ```

3. Ejecuta el script de instalación:
    ```sh
    cd instalacion
    install.bat
    ```

   El script `install.bat` instalará automáticamente los requisitos del proyecto y configurará Mask R-CNN.

### Método 2: Instalación Manual

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tu_usuario/Deteccion_Insectos_TFG.git
    cd Deteccion_Insectos_TFG
    ```

2. Se recomienda crear y activar un entorno virtual con conda como se ha hecho en este caso:
    ```sh
    conda create --name deteccion_insectos python=3.7.11
    conda activate deteccion_insectos
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

4. Configura Mask R-CNN:
    ```sh
    cd mrcnn
    python setup.py install
    ```

   Este paso instalará y configurará la biblioteca Mask R-CNN necesaria para el proyecto.

## Uso

Para ejecutar la aplicación, primero asegúrate de tener todas las dependencias instaladas y el entorno configurado. Luego, webapp  utiliza el siguiente comando:
    
    python app.py

    
## Estructura del Proyecto

- `Programacion_MaskRCNN/`: Contiene los scripts de entrenamiento y evaluación del modelo.
- `webapp/`: Contiene el código fuente de la aplicación web.
- `instalacion/`: Contiene los scripts de instalación y configuración.
- `LICENSE`: Licencia del proyecto.
- `README.md`: Archivo con información del proyecto.
- `anexos.pdf` y `memoria.pdf`: Documentación del proyecto.

## Contribución

Las contribuciones son bienvenidas. Para contribuir, por favor sigue los siguientes pasos:

1. Haz un fork del proyecto.
2. Crea una nueva rama (`git checkout -b feature/nueva-caracteristica`).
3. Realiza los cambios necesarios y haz commit (`git commit -am 'Añadir nueva característica'`).
4. Envía tus cambios (`git push origin feature/nueva-caracteristica`).
5. Crea un Pull Request.

## Posibles Mejoras

- **Optimización del Modelo**: Mejorar la precisión del modelo ajustando hiperparámetros y utilizando técnicas de aumento de datos.
- **Interfaz de Usuario**: Desarrollar una interfaz de usuario más intuitiva y amigable.
- **Soporte para Más Tipos de Insectos**: Ampliar el modelo para detectar y clasificar más tipos de insectos.
- **Despliegue en la Nube**: Implementar el proyecto en una plataforma en la nube para facilitar el acceso y la escalabilidad.
- **Documentación**: Ampliar y mejorar la documentación del proyecto, incluyendo tutoriales y guías paso a paso.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Para más detalles, consulta el archivo [LICENSE](LICENSE).

| Permissions | Limitations | Conditions |
| --- | --- | --- |
| ✔️ Commercial use | ❌ Liability | ℹ️ License and copyright notice |
| ✔️ Modification | ❌ Warranty |  |
| ✔️ Distribution |  |  |
| ✔️ Private use |  |  |
