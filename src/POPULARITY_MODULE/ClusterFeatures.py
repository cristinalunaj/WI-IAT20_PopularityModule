import os
import pandas as pd
import numpy as np
import src.metrics.metrics_clustering as metrics
import sklearn
from src.BaseArgs import ClusteringCleanerArgs
import src.utils.loader as loader

class ClusterFeatures():
    def __init__(self, input_dir, root_path_MTCNN_results, output_dir,program_name,program_participants, root_input_dir_parameters,
                 input_path_models,input_path_parameters, first_param_name = "eps", second_param_name ="min_samples", quality_metric="silhouette"):
        self.program = program_name
        self.program_participants = program_participants
        self.root_input_dir_parameters = root_input_dir_parameters
        self.quality_metric = quality_metric
        self.input_dir = input_dir
        self.first_param_name = first_param_name
        self.second_param_name = second_param_name
        self.root_input_path_embs = os.path.join(root_path_MTCNN_results, program_name, "embeddings_sum")
        self.root_input_path_bbox = os.path.join(root_path_MTCNN_results, program_name, "boundingboxes_sum")
        self.root_input_path_mtcnn = os.path.join(root_path_MTCNN_results, program_name, "mtcnn_debug_sum")
        self.input_path_models = input_path_models
        self.input_path_parameters = input_path_parameters



    def extract_clt_features(self,participant):
        """

        :param participant:
        :return:
        """
        #Load best parameters per participant
        input_path_parameters_participant = os.path.join(self.input_path_parameters, participant)
        input_path_models_participant = os.path.join(self.input_path_models, participant)
        _, _, clt_embs, embs_total_labels = loader.load_bboxes_and_embs(
            self.root_input_path_bbox, self.root_input_path_embs, [participant])
        df_global_features = pd.DataFrame([], columns=["id", "num_clts", "avg_silhouette", "avg_intra_clt_dist", "avg_inter_clt_dist", "faces_in_noise_clt", "num_core_samples"])
        df_individual_clt_features = pd.DataFrame([], columns=["id", "clt_id", "n_faces",
                               "avg_silhouette","std_silhouette",
                               "avg_intra_clt_dist", "std_intra_clt_dist",
                               "avg_inter_clt_dist", "std_inter_clt_dist",
                               "num_core_samples"])
        try:
            first_param,snd_param = loader.load_best_parameters_new(input_path_parameters_participant, self.quality_metric)
            #Load cluster
            path_cluster = os.path.join(input_path_models_participant,"cluster_"+self.first_param_name+"_"+str(first_param)+"_"+
                                        self.second_param_name+"_"+str(snd_param)+".sav")
            clt = loader.load_cluster(path_cluster)
            clt_labels = clt.labels_
            #clt_core_samples = clt.components_
            n_core_samples = clt.core_sample_indices_
            labels_core_samples = clt_labels[clt.core_sample_indices_]
            labels_ID = list(set(np.unique(clt_labels)).difference([-1]))
            intra_clust_dists, inter_clust_dists = metrics.intra_cluster_distance(clt_embs, clt_labels)
            silh_samples = sklearn.metrics.silhouette_samples(clt_embs, clt_labels)
            #GLOBAL FEATURES:
            faces_in_noise_clt = np.where(clt_labels==-1)
            silh_global = np.nanmean(silh_samples)
            avg_intra_clt_dist = np.nanmean(intra_clust_dists)
            avg_inter_clt_dist = np.nanmean(inter_clust_dists)
            df_global_features = pd.DataFrame([[participant, len(labels_ID),silh_global,avg_intra_clt_dist,avg_inter_clt_dist,len(faces_in_noise_clt[0]),len(n_core_samples)]], columns=["id", "num_clts", "avg_silhouette", "avg_intra_clt_dist", "avg_inter_clt_dist", "faces_in_noise_clt", "num_core_samples"])

            # PER CLUSTER FEATURES:
            for label in labels_ID:
                index_current_label = np.where(clt_labels==label)
                df_individual_clt_features = df_individual_clt_features.append(pd.DataFrame([[participant, label, len(index_current_label[0]),
                   np.nanmean(silh_samples[index_current_label]),np.nanstd(silh_samples[index_current_label]),
                   np.nanmean(intra_clust_dists[index_current_label]),np.nanstd(intra_clust_dists[index_current_label]),
                   np.nanmean(inter_clust_dists[index_current_label]),np.nanstd(inter_clust_dists[index_current_label]),
                   len(np.where(labels_core_samples == label)[0])]],
                   columns=["id", "clt_id", "n_faces",
                               "avg_silhouette","std_silhouette",
                               "avg_intra_clt_dist", "std_intra_clt_dist",
                               "avg_inter_clt_dist", "std_inter_clt_dist",
                               "num_core_samples"]))
        except FileNotFoundError:
            print("PASS")
            pass
        return df_global_features, df_individual_clt_features






if __name__ == "__main__":
    #Parse arguments
    clustering_cleaner_args_obj = ClusteringCleanerArgs()
    args = clustering_cleaner_args_obj.parse()

    # Get input path
    if (args.assignation_type == "x"):
        args.qualityTh_valid_cluster = "0"

    # Create complete cluster per program
    folder_2_save_data = ""
    if (args.with_previous_common_clusters):
        folder_2_save_data += "COMMON"
    if (args.with_previous_individual_clusters):
        if ("COMMON" in folder_2_save_data):
            folder_2_save_data = "IND_" + folder_2_save_data
        else:
            folder_2_save_data = "IND"


    input_dir = os.path.join(args.root_input_dir, "Cfacial_clustering_results", args.dataset, folder_2_save_data,
                             args.assignation_method,
                             "assignation_" + args.assignation_type,
                             "clustering_th_" + str(args.qualityTh_valid_cluster),
                             "cluster_filtering_assignation")
    root_input_dir_parameters = os.path.join(args.root_input_dir, "Cfacial_clustering_results", args.dataset)


    output_dir = args.output_dir
    # Extract programs & participants
    extension = os.listdir(args.program_participants_folder)[0].split(".")[-1] \
        if ("." in os.listdir(args.program_participants_folder)[0]) else ''
    programs = sorted(os.listdir(input_dir)) if (
            args.program_name == None or args.program_name == 'None') else [args.program_name]

    participants_per_program = {program: list(set(['OTHER'] + loader.load_list_of_tertulianos(
                                                      args.program_participants_folder, program,
                                                      extension))) for program in programs}

    replace_underScore = True
    for program in programs:
        #Create objects per program:
        try:
            participants_per_program[program].remove("OTHER")
        except ValueError:
            pass
        input_path_mtcnn = os.path.join(args.root_path_MTCNN_results, program, "mtcnn_debug_sum")
        input_path_embs = os.path.join(args.root_path_MTCNN_results, program, "embeddings_sum")
        input_path_bbox = os.path.join(args.root_path_MTCNN_results, program, "boundingboxes_sum")
        input_path_models = os.path.join(args.root_input_dir, "Cfacial_clustering_results", args.dataset, "cluster_models",
                                         program)
        input_path_parameters = os.path.join(args.root_input_dir, "Cfacial_clustering_results", args.dataset,
                                             "cluster_graphics", "parameters", program)
        cleaner_obj = ClusterFeatures(input_dir, args.root_path_MTCNN_results, output_dir, program,
                                  participants_per_program[program],
                                  root_input_dir_parameters, input_path_models, input_path_parameters,
                                  args.first_param_name, args.second_param_name, args.quality_metric)
        for participant in sorted(participants_per_program[program]):
            if (os.path.exists(os.path.join(args.output_dir, participant.replace(" ", "_") + "_cluster_features_individual.csv"))):
                continue
            df_global_features, df_individual_clt_features = cleaner_obj.extract_clt_features(participant)
            df_global_features.to_csv(os.path.join(args.output_dir, participant.replace(" ", "_") + "_cluster_features_global.csv"), sep=";", index=False)
            df_individual_clt_features.to_csv(os.path.join(args.output_dir, participant.replace(" ", "_") + "_cluster_features_individual.csv"), sep=";",index=False)


#PARAMETERS FOR RUNNING
# --root-input-dir
# /mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/supervised_results/BASELINE_OFICIAL_LN24H_SOLOIND
# --add-noise-class
# False
# --assignation-type
# x
# --qualityTh-valid-cluster
# 0.0
# --assignation-method
# numImgs
# --dataset
# Google
# --program-participants-folder
# /mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/rttm_INFO/FACEREF/participants
# --output-dir
# /mnt/RESOURCES2/POPULARITY_EXPERIMENTS/FEATURES/CLUSTER
# --with-previous-individual-clusters
# --first-param-name
# eps
# --second-param-name
# min_samples
# --quality-metric
# silhouette
# --root-path-MTCNN-results
# /mnt/RESOURCES/DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN
# --program-name
# TOTAL