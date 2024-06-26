@echo off

rem Instalar dependencias
pip install -r "%~dp0requirements.txt"

rem Instalar mrcnn
python "%~dp0setup.py" install

rem Espera a que el usuario vea el resultado antes de cerrar la ventana
pause
