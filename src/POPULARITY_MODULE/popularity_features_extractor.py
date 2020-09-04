from pytrends.request import TrendReq
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import os
import time
import ast

#DE-NORMALIZATION: https://es.slideshare.net/ShahanAliMemon/google-trends-denormalized-175072435
#NORMALIZE BY POPULATION: https://medium.com/@dhirajkhanna/normalizing-google-trends-7cbabf09f8f




def get_query_features_GoogleTrends(participant_name, hl = 'en-US', geo = "", gprop=""):
    #https://towardsdatascience.com/telling-stories-with-google-trends-using-pytrends-in-python-a11e5b8a177
    #download imgs agosto 2020
    #SPAIN
    # pytrends = TrendReq(hl='es-ES', tz=360, geo="ES")
    # df_by_region_SPAIN = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True, inc_geo_code=True)
    #WORLD
    # '2020-06-01 2017-06-01' example '2016-12-14 2017-01-25'
    pytrends = TrendReq(hl=hl, tz=360, geo=geo)
    pytrends.build_payload([participant_name], cat=0, timeframe='2017-06-01 2020-06-01', geo=geo, gprop=gprop)
    df_over_time = pytrends.interest_over_time()
    query_popularity_by_country = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=True)
    query_popularity_by_country["COUNTRY"] = query_popularity_by_country.index
    query_popularity_by_country.index = range(len(query_popularity_by_country))
    query_popularity_by_country = query_popularity_by_country.sort_values([participant_name], ascending=False)
    #dict_related_topic = pytrends.related_topics()
    #suggestions = pytrends.suggestions(participant_name)
    #N_COUNTRIES:
    # result_higher50 = query_popularity_by_country[query_popularity_by_country[participant_name]>=50]
    # result_lower50 = query_popularity_by_country[query_popularity_by_country[participant_name] < 50]
    return query_popularity_by_country


def get_suggestions_and_topics(participant_name, hl = 'en-US', geo = "", gprop=""):
    pytrends = TrendReq(hl=hl, tz=360, geo=geo)
    pytrends.build_payload([participant_name], cat=0, timeframe='2017-06-01 2020-06-01', geo=geo, gprop=gprop)
    dict_related_topic = pytrends.related_topics()
    suggestions = pytrends.suggestions(participant_name)
    return dict_related_topic, suggestions



def get_query_features_GoogleTrends_comparison(list_participant_names, country = 'en-US', geo = ""):
    #https://towardsdatascience.com/telling-stories-with-google-trends-using-pytrends-in-python-a11e5b8a177
    #download imgs agosto 2020
    #SPAIN
    # pytrends = TrendReq(hl='es-ES', tz=360, geo="ES")
    # df_by_region_SPAIN = pytrends.interest_by_region(resolution='REGION', inc_low_vol=True, inc_geo_code=True)
    #WORLD
    # '2020-06-01 2017-06-01' example '2016-12-14 2017-01-25'
    pytrends = TrendReq(hl=country, tz=360, geo = geo)
    pytrends.build_payload(list_participant_names, cat=0, timeframe='2017-06-01 2020-06-01', geo=geo, gprop='')
    #df_over_time = pytrends.interest_over_time()
    query_popularity_by_country = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=True)
    query_popularity_by_country["COUNTRY"] = query_popularity_by_country.index
    query_popularity_by_country.index = range(len(query_popularity_by_country))
    return query_popularity_by_country#, df_over_time


def get_single_img_features(img_path, mtcnn_path, id):
    # Img area:
    loaded_img = np.asarray(Image.open(img_path))
    img_area = loaded_img.shape[0]*loaded_img.shape[1]
    # Number of faces
    loaded_mtcnn = np.load(mtcnn_path, allow_pickle=True)
    faces = loaded_mtcnn["arr_0"]
    n_faces = len(faces)
    faces_data_df = pd.DataFrame([], columns=["id", "img_path", "face_index", "parent_img_area","confidence", "relative_size", "third_rule_x","third_rule_y"])
    face_index = 0
    for face in faces:
        x_topL, y_topL, width, height = face['box']
        face_area = width*height
        confidence = face["confidence"]
        relative_size = (face_area/img_area)*100
        center_face_xy = [(x_topL+(width/2)),(y_topL+(height/2))]
        third_rule_xy = get_position_third_rule(loaded_img.shape[1], loaded_img.shape[0], center_face_xy)
        faces_data_df = faces_data_df.append(pd.DataFrame([[id, img_path,face_index,img_area,confidence,relative_size,third_rule_xy[0],third_rule_xy[1]]], columns=["id","img_path", "face_index", "parent_img_area","confidence", "relative_size", "third_rule_x","third_rule_y"]))
        # cv2.circle(loaded_img,(int(center_face_xy[0]), int(center_face_xy[1])),1,(0,0,255),2)
        # cv2.imshow("img",loaded_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        face_index+=1
    return faces_data_df, n_faces


def get_position_third_rule(img_width, img_heigth, center_face_xy):
    """
         (0,0)---------------------------> img_width
            |   00    |   10   |   20   |
            |---------|--------|--------|
            |   01    |   11   |   21   |
            |---------|--------|--------|
            |   02    |   12   |   22   |
            |---------|--------|--------|
            v
            img_heigth

    :param img_width:
    :param img_heigth:
    :param center_face_xy:
    :return:
    """
    return (int(center_face_xy[0]/(img_width/3)), int(center_face_xy[1]/(img_heigth/3)))


def get_INDcluster_features(path_INDclusters, path_metrics_clusters, participant_name):
    print("to do")
    #Number of clusters
    path_participant_INDclusters = os.path.join(path_INDclusters, participant_name.replace("_"," "), "faces")
    n_clusters = len(os.listdir(path_participant_INDclusters))
    #Load metrics dataframe
    quality_df = pd.read_csv(path_metrics_clusters, header=0, sep=";")
    quality_df.index = quality_df.ID_name
    #Participant cluster quality:
    avg_silhouette = quality_df.loc[participant_name.replace("_", " "), "avg_silhouette"]
    intra_clt_dist = quality_df.loc[participant_name.replace("_", " "), "avg_intra_cluster_dist"]
    inter_clt_dist = quality_df.loc[participant_name.replace("_", " "), "avg_inter_cluster_dist"]
    imgs_per_cluster = []
    for cluster in os.listdir(path_participant_INDclusters):
        clt_path = os.path.join(path_participant_INDclusters, cluster)
        n_imgs_clt = len(os.listdir(clt_path))
        imgs_per_cluster.append(n_imgs_clt)
    return n_clusters, avg_silhouette, intra_clt_dist, inter_clt_dist, imgs_per_cluster



def check_min_GoogleTrends(participants, out_path_features):
    min_popularity = 250*100
    min_popularity_id = ""
    df_popularity = pd.DataFrame([], columns=["id", "sum_popularity", "first_country"])
    for participant in participants:
        path_participant = os.path.join(out_path_features, participant+"_global.csv")
        df_participant = pd.read_csv(path_participant, header=0, sep=";")
        sum_popularity = df_participant[participant.replace("_", " ")].sum()
        print(participant,": ", str(sum_popularity))
        df_popularity = df_popularity.append(pd.DataFrame([[participant, sum_popularity, df_participant.iloc[0]["COUNTRY"]]], columns=["id", "sum_popularity", "first_country"]))
        if(sum_popularity<min_popularity):
            min_popularity_id = participant
            min_popularity = sum_popularity
    print("MINIMUM: ", min_popularity_id,": ", str(min_popularity))
    df_popularity = df_popularity.sort_values(by="sum_popularity", ascending=False)
    median = df_popularity["sum_popularity"].median()
    df_popularity.to_csv(os.path.join(out_path_features, "TOTAL_sum_single_popularity.csv"), sep=";", header=True, index=False)


def join_global_features(participant, df_imgs, df_clt_global, df_clt_ind, df_GTrends, df_labels):
    #IMAGE:
    df_imgs_avg = df_imgs.mean()
    df_imgs_std = df_imgs.std()
    df_img_total = pd.DataFrame([[len(df_imgs),df_imgs_avg["confidence"],df_imgs_std["confidence"],
                                 df_imgs_avg["relative_size"],df_imgs_std["relative_size"],
                                 df_imgs_avg["third_rule_x"], df_imgs_std["third_rule_x"],
                                 df_imgs_avg["third_rule_y"], df_imgs_std["third_rule_y"],
                                 ]], columns=["n_faces","avg_confidence_faces", "std_confidence_faces",
                                             "avg_relativeSize_faces", "std_relativeSize_faces",
                                             "avg_thirdRule_x", "std_thirdRule_x",
                                             "avg_thirdRule_y", "std_thirdRule_y"])
    #CLUSTER
    df_clt_global["avg_imgs_clt"] = df_clt_ind["n_faces"].mean()
    df_clt_global["avg_std_silhouette"] = df_clt_ind["std_silhouette"].mean()
    df_clt_global["avg_std_intra_clt_dist"] = df_clt_ind["std_intra_clt_dist"].mean()
    df_clt_global["avg_std_inter_clt_dist"] = df_clt_ind["std_inter_clt_dist"].mean()
    df_clt_global["avg_n_core_samples"] = df_clt_ind["num_core_samples"].mean()
    df_clt_global["std_n_core_samples"] = df_clt_ind["num_core_samples"].std() if(df_clt_global["num_clts"][0]>1) else 0
    #GTRENDS
    participant_popularity = df_GTrends.loc[df_GTrends['id'] == participant]["sum_popularity"].values[0]

    #LABELS:
    index_user = df_labels.index[df_labels['Nombre_personaje'] == participant].tolist()[0]
    n_true_imgs_dw = len(ast.literal_eval(df_labels.iloc[index_user]["TP"]) + ast.literal_eval(df_labels.iloc[index_user]["FN"]))
    label = n_true_imgs_dw
    if(n_true_imgs_dw > 50):
        label = 1
    else:
        label = 0
    #JOIN
    df_col = pd.concat([df_img_total, df_clt_global], axis=1)
    df_col["GTrends_popularity"] = participant_popularity
    df_col["label"] = label
    return df_col, n_true_imgs_dw


if __name__ == "__main__":
    # IMAGE FEATURES
    train = False
    root_path_face_diarization = "/home/cristinalunaj/PycharmProjects/RTVE_2018_MM_DIARIZATION"
    if(train):
        #IMG
        path_imgs = "/mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/DATASET_GOOGLE_IMGS/download_name"
        path_mtcnn = "/mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN/TOTAL/mtcnn_debug_100"
        out_path_img_features = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/IMAGES/train/"
        #CLT
        out_dir_clt = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/CLUSTER/train"
        program_participants_folder = "/mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/rttm_INFO/FACEREF/participants"
        path_mtcnn = "/mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN"
        path_mtcnn_imgs = "/mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN/TOTAL/mtcnn_debug_100"
        root_input_dir = "/mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/supervised_results/BASELINE_OFICIAL_LN24H_SOLOIND"
        #GTRENDS
        output_path_single_df = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/train/SINGLE_ID_global_all"
        # output_path_single_df_spain = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/train/SINGLE_ID_SPAIN_all"
        # out_path_single_df_suggestions_dict_spain = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/train/SUGGESTIONS_spain"
        # out_path_single_df_suggestions_dict_world = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/train/SUGGESTIONS_global"
        #LABELS:
        path_labels = "/mnt/RESOURCES2/ETIQUETADO_RTVE/CLUSTERS_TOTAL_RTVE2018_TEST_globalMAX/TOTAL/x_eps_0_min_samples_0/CUESTIONARIOS/total_cuestionarios.csv"
        #JOIN:
        out_path_total_features = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/DATASETS/train/summary_features_participants_classification.csv"
    else:
        #IMG
        path_imgs = "/mnt/RESOURCES/DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/GENERATED_ME/DATASET_GOOGLE_IMGS/download_name"
        path_mtcnn = "/mnt/RESOURCES/DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/GENERATED_ME/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN"
        path_mtcnn_imgs = "/mnt/RESOURCES/DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/GENERATED_ME/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN/TOTAL/mtcnn_debug_100"
        out_path_img_features = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/IMAGES/test/"
        #CLT
        out_dir_clt = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/CLUSTER/test"
        program_participants_folder = "/mnt/RESOURCES/DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/rttm_INFO/FACE/participants"
        root_input_dir = "/mnt/RESOURCES/DATASET_RTVE_2020/GOOGLE_RTVE_2020/dev/GENERATED_ME/supervised_results/BASELINE_OFICIAL_RTVE2020_globalMax"
        #GTRENDS
        output_path_single_df = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/test/SINGLE_ID_global_all"
        # output_path_single_df_spain = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/test/SINGLE_ID_SPAIN_all"
        # out_path_single_df_suggestions_dict_spain = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/test/SUGGESTIONS_spain"
        # out_path_single_df_suggestions_dict_world = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/GOOGLE_TRENDS/test/SUGGESTIONS_global"
        #LABELS:
        path_labels = "/mnt/RESOURCES2/ETIQUETADO_RTVE/CLUSTERS_TOTAL_RTVE2020_globalMAX/TOTAL/x_eps_0_min_samples_0/RTVE2020_Ricardo/CUESTIONARIOS/rtve2020_Ricardo_TOTAL.csv"
        #JOIN:
        out_path_total_features = "/mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/DATASETS/test/summary_features_participants_classification.csv"
    #CREATE FOLDERS:
    os.makedirs(out_path_img_features, exist_ok=True)
    os.makedirs(out_dir_clt, exist_ok=True)
    os.makedirs(output_path_single_df, exist_ok=True)
    os.makedirs(out_path_total_features.rsplit("/", 1)[0], exist_ok=True)

    list_participants = os.listdir(path_mtcnn_imgs)
    print("EXTRACTING IMAGE FEATURES ...")
    program = "TOTAL"
    for id in list_participants:
        mtcnn_id_path_npz = os.path.join(path_mtcnn, id)
        df_per_img = pd.DataFrame([], columns=["id", "img_path", "face_index", "parent_img_area", "confidence",
                                               "relative_size", "third_rule_x", "third_rule_y"])
        if(os.path.exists(os.path.join(out_path_img_features, id.replace(" ", "_") + "_img_features.csv"))):
            continue
        for mtcnn_npz in os.listdir(mtcnn_id_path_npz):
            path_id_npz = os.path.join(mtcnn_id_path_npz, mtcnn_npz)
            path_img = os.path.join(path_imgs, id, mtcnn_npz.split(".")[0]+".png")
            img_df, n_faces = get_single_img_features(path_img, path_id_npz, id)
            df_per_img = df_per_img.append(img_df)
        df_per_img.to_csv(os.path.join(out_path_img_features, id.replace(" ", "_") + "_img_features.csv"), sep=";",index=False)
    #get real/model features summary ...

    # CLUSTER FEATURES

    print("EXTRACTING CLUSTER FEATURES ...")
    path_2_virtualenv = "/home/cristinalunaj/.virtualenvs/RTVE_2018_MM_DIARIZATION/bin/activate"
    command = ". " + path_2_virtualenv + " ;python3 " + root_path_face_diarization + "/src/FACE_DIARIZATION/B_GooglePopularity/ClusterFeatures.py \
                    --root-input-dir "+root_input_dir+" \
                    --add-noise-class False \
                    --assignation-type x \
                    --qualityTh-valid-cluster 0.0 \
                    --assignation-method numImgs \
                    --dataset Google \
                    --program-participants-folder "+program_participants_folder+" \
                    --output-dir "+out_dir_clt+" \
                    --with-previous-individual-clusters \
                    --first-param-name eps \
                    --second-param-name min_samples \
                    --quality-metric silhouette \
                    --root-path-MTCNN-results "+path_mtcnn+"\
                    --program-name TOTAL"
    exit_code = os.system(command)

    #GOOGLE TRENDS - FEATURES
    print("EXTRACTING GOOGLE-TRENDS FEATURES ...")
    for i in range(len(list_participants)):
        name = list_participants[i]
        print("DOING: ", name)
        #GLOBAL IMPACT
        if(not os.path.exists(os.path.join(output_path_single_df, name.replace(" ", "_")+"_global.csv"))):
            df_name = get_query_features_GoogleTrends(name.replace("_", " "))
            df_name.to_csv(os.path.join(output_path_single_df, name.replace(" ", "_")+"_global.csv"), sep=";", index=False)
            time.sleep(30)
    check_min_GoogleTrends(list_participants, output_path_single_df)

    #JOIN FEATURES:
    df_total = pd.DataFrame([], columns=['n_faces', 'avg_confidence_faces', 'std_confidence_faces', 'avg_relativeSize_faces', 'std_relativeSize_faces', 'avg_thirdRule_x', 'std_thirdRule_x', 'avg_thirdRule_y', 'std_thirdRule_y', 'id', 'num_clts', 'avg_silhouette', 'avg_intra_clt_dist', 'avg_inter_clt_dist', 'faces_in_noise_clt', 'num_core_samples', 'avg_imgs_clt', 'avg_std_silhouette', 'avg_std_intra_clt_dist', 'avg_std_inter_clt_dist', 'avg_n_core_samples', 'std_n_core_samples', 'GTrends_popularity', 'label'])
    max_n_imgs = 0
    labels = []
    for id in list_participants:
        path_img_features = os.path.join(out_path_img_features, id.replace(" ", "_") + "_img_features.csv")
        df_imgs = pd.read_csv(path_img_features,header=0, sep=";")

        path_clt_ind = os.path.join(out_dir_clt, id.replace(" ", "_") + "_cluster_features_individual.csv")
        path_clt_global = os.path.join(out_dir_clt, id.replace(" ", "_") + "_cluster_features_global.csv")
        df_clt_global = pd.read_csv(path_clt_global, header=0, sep=";")
        df_clt_ind = pd.read_csv(path_clt_ind, header=0, sep=";")

        path_GTrends = os.path.join(output_path_single_df, "TOTAL_sum_single_popularity.csv")
        df_GTrends_global = pd.read_csv(path_GTrends, header=0, sep=";")

        df_labels = pd.read_csv(path_labels, header=0, sep=";")
        df_participant,n_imgs = join_global_features(id, df_imgs, df_clt_global,df_clt_ind, df_GTrends_global, df_labels)
        df_total = df_total.append(df_participant)
        labels.append(n_imgs)
        if(n_imgs>max_n_imgs):
            max_n_imgs = n_imgs
    df_total.to_csv(out_path_total_features, index=False, sep=",", header=True)
    print("MAX N IMGS: ", str(max_n_imgs))


