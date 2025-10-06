#!/bin/bash

# substituir 1001:1001 se o uid/gid do spark for outro
sudo mkdir -p ./shared/utils/processed_output ./shared/utils/datasets
sudo chown -R 1001:1001 ./shared/utils/processed_output ./shared/utils/datasets
sudo chmod -R 0777 ./shared/utils/processed_output ./shared/utils/datasets

# garantir subdiretório usado pelos executores do Spark
sudo mkdir -p ./shared/utils/processed_output/spark_tmp
sudo chown -R 1001:1001 ./shared/utils/processed_output/spark_tmp
sudo chmod -R 0777 ./shared/utils/processed_output/spark_tmp