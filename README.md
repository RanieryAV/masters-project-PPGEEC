# masters-project-PPGEEC

## Project CAPTAIMA (ENGLISH)
<p><b>C</b>lassify <b>A</b>nd <b>P</b>redic<b>T</b> <b>A</b>bnormal <b>I</b>nstances of <b>M</b>aritime <b>A</b>ctivity - <b>CAPTAIMA</b>

***
***

## Projeto CAPTAIMA (PORTUGUESE)
<p><b>C</b>lassific<b>A</b>r e <b>P</b>redizer ins<b>T</b>âncias <b>A</b>normais de at<b>I</b>vidade <b>M</b>arítim<b>A</b> - <b>CAPTAIMA</b></p>

***
***

## DESCRIPTION (ENGLISH)
<h3>Research Plan's Theme: CLASSIFICATION AND PREDICTION OF MARITIME ANOMALIES USING MACHINE LEARNING IN GEOGRAPHIC INFORMATION SYSTEMS (GIS)</h3>
<h3>- Federal University of Ceará (Sobral Campus)</h3>
<h3>- Graduate Program in Electrical and Computer Engineering (PPGEEC)</h3>

***
***

## DESCRIÇÃO (PORTUGUESE)
<h3>- Tema do plano de pesquisa: CLASSIFICAÇÃO E PREDIÇÃO DE ANOMALIAS MARÍTIMAS POR MEIO DO USO DE APRENDIZAGEM DE MÁQUINA EM SISTEMAS DE INFORMAÇÃO GEOGRÁFICA (SIG)</h3>
<h3>- Universidade Federal do Ceará (Campus Sobral)</h3>
<h3>- Programa de Pós-Graduação em Engenharia Elétrica e de Computação (PPGEEC)</h3>


# Setup instructions
1. Clone this repository
    ```sh
    git clone https://github.com/yourusername/masters-project-PPGEEC.git
    cd masters-project-PPGEEC
    ```
2. Run the `install_dependencies.sh` script to install the required dependencies.
    ```sh
    chmod +x scripts/install_dependencies.sh
    ./scripts/install_dependencies.sh
    ```
3. Run the `set_up_folder_permissions.sh` script to start the Data Processing API and configure crucial directories.
    ```sh
    chmod +x scripts/set_up_folder_permissions.sh
    ./scripts/set_up_folder_permissions.sh
    ```
4. Place general dataset files into `shared/utils/datasets/`.
 - TODO: Specify dataset files and their sources.
5. Run the `docker-compose-infra.yml` file to start the necessary infrastructure services.
    ```sh
    docker compose -f docker-compose-infra.yml up -d --build
    ```
6. Run the `docker-compose-services.yml` file to start the essential services.
    ```sh
    docker compose -f docker-compose-apps.yml up -d --build
    ```

NOTES:
- It was discovered that in order for the local data processing API to be able to send Spark jobs to the Spark containers, the Driver host must also be a container. Insisting on using the local API would require additional configurations that are not worth the effort at this moment (such as using extra softwares, like Lily, which would increase function verbosity and add an extra failure point).
    - Consequence: Only the containerized data processing API can send Spark jobs to the Spark containers.
    - That leaves the local APIs for debugging purposes only.