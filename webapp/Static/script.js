

document.addEventListener('DOMContentLoaded', function () {
    const namespace = '/entrenamiento';
    const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);
    



    socket.on('update_epoch', function(data) {
        // Si es la primera actualización, ocultar el mensaje de cargando
        document.getElementById('epoch-info').classList.remove('hidden');
        document.getElementById('loading-info').classList.add('hidden');
        document.getElementById('epoch-info').textContent = `Epoca: ${data.current_epoch} de ${data.total_epochs}`;
    });

    socket.on('update_batch', function(data) {
        // Si es la primera actualización, ocultar el mensaje de cargando
        document.getElementById('loading-info').classList.add('hidden');
        document.getElementById('batch-info').classList.remove('hidden');
        if (data.val_loss) {
            document.getElementById('batch-info').textContent = `Step: ${data.current_batch} de ${data.total_batches}, Perdida: ${data.loss}, Perdida de validacion: ${data.val_loss}`;
        } else {
            document.getElementById('batch-info').textContent = `Step: ${data.current_batch} de ${data.total_batches}, Perdida: ${data.loss}`;
        }
    });

    socket.on('training_complete', function(data) {
        // Se genera el enlace de descarga cuando el entrenamiento está completo
        var downloadLink = document.createElement('a');
        downloadLink.href = `/download/${data.file}`;
        downloadLink.textContent = 'Descargar Modelo Entrenado';
        downloadLink.download = data.file; // Esto sugerirá el nombre en el diálogo de guardar como

        var downloadLinkArea = document.getElementById('download-link-area');
        downloadLinkArea.innerHTML = ''; // Limpia cualquier enlace previo
        downloadLinkArea.appendChild(downloadLink);
    });
    
    socket.on('update_data', function(data) {
        console.log('Elementos actualizados con los datos recibidos');
    
        // Llenar los elementos con los datos recibidos y remover la clase 'hidden' para mostrarlos
        document.getElementById('filename-info').textContent = 'Nombre de archivo: ' + data.filename;
        document.getElementById('filename-info').classList.remove('hidden');
    
        document.getElementById('algorithm-info').textContent = 'Algoritmo: ' + data.algorithm;
        document.getElementById('algorithm-info').classList.remove('hidden');
    
        document.getElementById('training-prop-info').textContent = 'Proporcion de entrenamiento: ' + data.proporcionEntrenamiento;
        document.getElementById('training-prop-info').classList.remove('hidden');
    
        document.getElementById('test-prop-info').textContent = 'Proporcion de test: ' + data.proporcionTest;
        document.getElementById('test-prop-info').classList.remove('hidden');
    
        document.getElementById('steps-info').textContent = 'Numero de steps: ' + data.steps;
        document.getElementById('steps-info').classList.remove('hidden');
    
        document.getElementById('model-info').textContent = 'Modelo ResNet: ' + data.Resnet;
        document.getElementById('model-info').classList.remove('hidden');
    
        document.getElementById('validation-steps-info').textContent = 'Steps de validacion: ' + data.validacion;
        document.getElementById('validation-steps-info').classList.remove('hidden');
    
        // Agregar un mensaje de registro en la consola del navegador
        console.log('Elementos mostrados tras actualización de datos');
    });
});


