#!/bin/bash

# Certifique-se de que as pastas existem
mkdir -p shared/utils/datasets
mkdir -p shared/utils/processed_output

# Ajusta dono para seu usuário (supondo UID e GID 1000)
sudo chown -R 1000:1000 shared/utils/datasets
sudo chown -R 1000:1000 shared/utils/processed_output

# Dá permissão de escrita/gravação/execução de diretórios
sudo chmod -R 775 shared/utils/datasets
sudo chmod -R 775 shared/utils/processed_output

