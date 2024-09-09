#!/bin/bash
# Iniciar o Orthanc em background
orthanc /etc/orthanc &

# Espera um pouco para o Orthanc estar pronto
sleep 10

# Executar o script de envio de DICOMs
python3 /app/scripts/send_dicom.py

# Manter o container rodando
tail -f /dev/null
