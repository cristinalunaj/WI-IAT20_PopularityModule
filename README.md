# WI-IAT20_PopularityModule

Experiments and code of the popularity module sent to the conference WI-IAT20.

REPOSITORY UNDER DEVELOPMENT !!!

### Description
This project contains scripts for annotating data in a weakly-supervised way based on the help of a clustering procedure. It also allows to download Google images for using them to train the popularity module, 
in charge of discern who is popular and who is not based on a set of features extracted from the clustering, the images downloaded and GoogleTrends.

Results of the paper are evaluated using the participant names provided by some of the videos of 2 corpus, accessible through a license for the competitions:
* RTVE2018DB: http://iberspeech2018.talp.cat/
* RTVE2020DB: http://catedrartve.unizar.es/albayzindatabases.html

### Installation
In order to replicate the experiments presented in WI-IAT20 conference, it is required to clone the repository and install the required libraries under requirements.txt. 
It is also highly recommended to create a virtual environment to work from it. 
Below, you can follow the installation steps to download all the files in the repository and install your own virtual environment for the project

```
mkdir WI_IAT20
git clone https://github.com/cristinalunaj/WI-IAT20_PopularityModule
cd WI_IAT20
python3 -m venv WI_IAT20_venv
source WI_IAT20_venv/bin/activate
pip install --upgrade pip
pip install .
cd ..
cd WI-IAT20_PopularityModule
pip install -r requierements.txt
sudo apt-get install chromium-chromedriver
```

#### Requirements

Apart from the Python packages installed previously, it is neccesary to install the following pre-requisites:
#####Installation of chrome & chrome-driver -> For downloading Google images
-Install/update Chrome: (https://www.google.com/chrome/)
-Install/update chromedriver according to your chrome version and OS (https://chromedriver.chromium.org/)
(check: https://www.google.com/chrome/)



### Usage 
** NOTE:For the Usage examples we will consider that the corpus data was downloaded under: ../DATASET_RTVE_2020/

The system contains several modules that can be used in a separate way. In this section, we will introduce some examples of use of each module 
as well as their results. Examples are illustrated for the case of using RTVE2020 dataset participants. 

1. This module generates an automatic list of participants per program as well as a complete list with all the participants of the programs (participants_complete.csv), as it can be seen
in an example under: data/datasets/DATASET_GOOGLE_IMGS/participants/participants_complete_rtve2020.csv.
    ```
    python3 DSPreparator.py
    --input-path-labels-rttm ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm
    --output-path-labels-rttm ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO/Amd_eval_adapted_rttm
    --output-path-participants-folder ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO
    --category FACE
    --programs-folder ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/video
    --semi-supervised True
    ```

2. This module download images from Google using Selenium. It has an offset of 5 images, so if 150 are required in many cases 155 will be provided except when there are errors.
IMPORTANT: This script does not run in second plane so, once it is launched it blocks the GUI.

    ```
    python3 GoogleQueryLauncher.py
    --participants-path ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO/FACE/participants_complete.csv
    --imgs-2-download 150 
    --chrome-driver-path .../chromedriver_linux64/chromedriver
    --output-dir WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/download_name
    --logs-path WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/URL_imgs_logs/rtve2020
    ```

3. Change structure of the dataset to save all the particiapants under a folder with name: 'TOTAL', that will be our program_name

    ```
    python3 RefactorDS.py
    --root-input-path WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/download_name
    --set-to-analyse Google
    --program-participants-folder ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO/FACE/participants
    --output-dir WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/refactor_DS
    ```

4. Detect faces in image and extract bounding boxes using MTCNN (https://pypi.org/project/mtcnn/). After that, pass the generated 
bounding boxes to FaceNet, extracting an embedding representation of tthe face, which locate the face in a latent space of 128-dimensions.
FaceNet weigths are saved in: data/models/pre_trained_models/face_embs_model/facenet_keras.h5.
In this case, the number of images processed will be 100 and the confidence required to faces extracted by MTCNN to be a face is: 98%.
    ```
    python3 FaceDetectorAndEncoder.py
    --face-detector MTCNN
    --root-input-folder WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/refactor_DS
    --program-name TOTAL
    --output-dir WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN
    --program-participants-folder ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO/FACE/participants
    --imgs-2-maintain 100
    --face-threshold 0.98
    --encoding-model WI-IAT20_PopularityModule/data/models/pre_trained_models/face_embs_model/facenet_keras.h5
    ```

5. Create DBSCAN clusters scanning from eps 0.5 - 12.0 in steps of 0.5. For each participant, 24 configurations are generated, from which parameters
like the silhouette are extracted in order to compare quality of each one, and select the best configuration.
    ```
    python3 ClusterScan.py
    --root-path-MTCNN-results WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN
    --program-participants-folder ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO/FACE/participants
    --program-name TOTAL
    --first-param-lower 0.5
    --first-param-upper 12
    --first-param-step 0.5
    --first-param-name eps
    --second-param-name min_samples
    --second-param-value [5]
    --metric euclidean
    --dataset Google
    --output-dir WI-IAT20_PopularityModule/data/results/supervised_results/BASELINE_OFICIAL_RTVE2020_globalMax
    --cluster-instance DBSCAN
    --individual-clusters
    --quality-metric silhouette
    ```

6. Once the best clustering configuration is chosen, this module selects the most likely cluster of representing the participant.
In this case,the criterion for selecting the most probable cluster is based on the number of images.   
    ```
    python3 IndividualClusterConsolidation.py
    --root-path-MTCNN-results WI-IAT20_PopularityModule/data/datasets/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN
    --program-participants-folder ../DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO/FACE/participants
    --program-name TOTAL
    --first-param-lower 0.5
    --first-param-upper 12
    --first-param-step 0.5
    --first-param-name eps
    --second-param-name min_samples
    --second-param-value [5]
    --metric euclidean
    --dataset Google
    --output-dir WI-IAT20_PopularityModule/data/results/supervised_results/BASELINE_OFICIAL_RTVE2020_globalMax
    --cluster-instance DBSCAN
    --consolidation-type numImgs
    --quality-metric silhouette
    ```

7. Popularity module...

... TO CONTINUE..

Diarization modules will be released soon, in the following repository: (TO DO)

### License

MIT