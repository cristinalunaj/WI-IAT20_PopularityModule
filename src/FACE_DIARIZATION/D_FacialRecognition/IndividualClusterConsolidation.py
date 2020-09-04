"""
    Create individual clusters per identity and consolidate the one that reaches the best quality and, thus, is most probable of being representing the
    participant.
	author: Cristina Luna, Ricardo Kleinlein
	date: 05/2020
	Usage:
		python3 IndividualClusterConsolidation.py
		--root-path-MTCNN-results ~/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN
        --program-participants-folder ~/test/
        --program-name LM-20170103
        --first-param-name eps
        --first-param-lower 0.5
        --first-param-upper 12
        --first-param-step 0.5
        --metric euclidean
        --cluster-instance DBSCAN
        --min-samples 10
        --metric euclidean
        --dataset Google
        --output-dir ~/supervised_results/BASELINE_OFICIAL_LN24H_v2
        --individual-clusters
        --quality-metric silhouette
        --with-previous-individual-clusters
        --with-previous-common-clusters
        --save-as-MTCNN
        --consolidation-type minDist
        --k 10
        --reference-parameters-path ~/supervised_results/BASELINE_OFICIAL_LN24H_v2/Cfacial_clustering_results/OCRCLUSTERING/cluster_graphics/parameters
        --reference-embs-path ~/supervised_results/BASELINE_OFICIAL_LN24H_v2/Cfacial_clustering_results/reference_dataset/x/assignation_x/clustering_th_x/cluster_filtering_assignation/LM-20170103/x_eps_x_min_samples_x/embeddings_sum
        --random-seed 42

	Options:
	    --root-path-MTCNN-results: Root path to the results obained by MTCNN (embs, bbox, mtcnn_debub, bbox_summ)
	    --program-participants-folder: Path to the folder with the .csv that contain the names of the
	    participants per program (See /AQuery_launcher/Aqeury_launcher_bingEngine.py
	    --first-param-name: Name of the first param as it appears in the DBSCAN and HDBSCAN libraries.
                 -DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
                 -HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan
                 [default: eps (DBSCAN param.)]
	    --first-param-lower: Lower value for first parameter [default: 0.5]. For DBSCAN this parameter is eps
	    --first-param-upper: Upper value for fisrt parameter [default: 12]. For DBSCAN this parameter is eps
	    --first-param-step: Step between 2 consecutive values of the first parameter. For DBSCAN this parameter is eps
	    --second-param-value: List with second param values [default: [10]]
	    --metric: Distance metric in DBSCAN [Options: euclidean, cosine, manhattan, l1, l2]
	    --dataset: Dataset reference name to extract clustering & save results(e.g Google, OCR, Google_OCR, Google_Program ...)
		--output-dir	Directory to save results in.By default, it will act as root folder and results will be saved in the
		path: output_dir/Cfacial_clustering_results/'dataset'/
		                                                |_cluster_models/'program'/
		                                                                    |_cluster_eps_x_minSamples_x.sav
		                                                |_cluster_graphics/parameters/'program'
		                                                                                |_(matrix/graphics...)
		--individual-clusters: True if we want a cluster per participant, else it creates a common cluster for all participants
        --quality-metric: Metric to use as quality and select the best cluster combination of parameters. Options ( cluster, v_meassure, silhouette, calinski_harabasz).
        --with-previous-individual-clusters: True if we did a previous individual cluster per participant, else False
        --with-previous-common-clusters: True if we did a previous common cluster with all the participants, else False.
        --save-as-MTCNN: True if we want save data as MTCNN output,else it assignate clusters to participants based on
         num. of images per cluster or by comparison with 'reference DS'
        --consolidation-type: How to consolidate individual clusters and select the chosen cluster. (Options: numImgs, minDist)
                 If numImgs -> Assign the cluster with the max. number of images as the best, and thus assign it to the participant
                 If minDist -> Assign the similarest cluster to the participant based on a second datset ('reference DS')
        --k: If consolidation-type=minDist is selected, it requires the k neighbours to compare and decide which is the best cluster
        --reference-parameters-path: If consolidation-type=minDist is selected, it requires the path where to find the -csv with the trust/non-trust clusters.
                 (default e.g. of path: args.output_dir/Cfacial_clustering_results/reference_dataset/cluster_graphics/parameters)
        --reference-embs-path:If consolidation-type=minDist is selected, it requires the path where to find the clusters of the datset to compare as well as their embeddings.
                 (default e.g. of path: args.output_dir/Cfacial_clustering_results/reference_dataset/x/assignation_x/clustering_th_x/cluster_filtering_assignation/PROGRAM/x_eps_x_min_samples_x/embeddings_sum)
        --random-seed: Random seed for generating splits of train/test [default: 42]
		-h, --help	Display script additional help
"""

import os, sys
import src.utils.loader as loader
import numpy as np
from src.BaseArgs import IndividualClusterConsolidatorArgs
import pandas as pd
import src.metrics.metrics_clustering as clt_metrics
import src.FACE_DIARIZATION.D_FacialRecognition.ClusterAssignation as cltAssignation
import sklearn
import random
import shutil
from sklearn.metrics.pairwise import euclidean_distances

class IndividualClusterConsolidator():

    def __init__(self, root_path_MTCNN_results, program_name, output_dir, program_participants, dataset,
                 first_param_name = "eps", second_param_name ="min_samples"):
        self.dataset = dataset
        self.first_param_name = first_param_name
        self.second_param_name = second_param_name
        # Input
        self.root_input_path = root_path_MTCNN_results
        self.program = program_name
        self.program_participants = program_participants
        # Output
        self.output_dir = output_dir

    def save_MTCNN_from_individual_clusters(self, participant, chosen_cluster_ID, clt, output_path_imgs):
        """

        :param participant:
        :param chosen_cluster_ID:
        :param clt:
        :param output_path_imgs:
        """
        cluster_labels = clt.labels_
        # Path embs & bbox:
        input_path_embs = os.path.join(self.root_input_path, self.program, "embeddings_sum")
        input_path_bbox = os.path.join(self.root_input_path, self.program, "boundingboxes_sum")

        # Load embs & bboxes for participants of program
        bbox_total_faces, bbox_total_labels, embs_total_faces, embs_total_labels = loader.load_bboxes_and_embs(
            input_path_bbox, input_path_embs, self.program_participants)
        # Calculate intra cluster distance of chosen cluster: ---- QUALITY METRICS ---------
        ids_nonNoise = np.where(cluster_labels == chosen_cluster_ID)
        new_cluster_labels = -1 * np.ones(shape=len(cluster_labels), dtype=int)
        new_cluster_labels[ids_nonNoise] = chosen_cluster_ID
        final_dict_assignation = {participant: [chosen_cluster_ID],
                                  "OTHER": [-1]}
        #Generate new MTCNN
        cltAssignation.save_faces_and_embeddings_from_clusters(output_path_imgs, embs_total_faces,
                                                               embs_total_labels,
                                                               bbox_total_faces, [chosen_cluster_ID, -1],
                                                               final_dict_assignation, new_cluster_labels,
                                                               label_4_unk = "OTHER", with_spaces = True, concatenate_if_exists=True)




    def save_clusters_from_individual_clusters(self, participant, chosen_cluster_ID, clt, output_path_imgs, debug=True,clt_index = 0):
        labels_ID = np.unique(clt.labels_)
        cluster_labels = clt.labels_
        # Path embs & bbox:
        input_path_embs = os.path.join(self.root_input_path, self.program, "embeddings_sum")
        input_path_bbox = os.path.join(self.root_input_path, self.program, "boundingboxes_sum")

        # Load embs & bboxes for participants of program
        bbox_total_faces, bbox_total_labels, embs_total_faces, embs_total_labels = loader.load_bboxes_and_embs(
            input_path_bbox, input_path_embs, self.program_participants)
        #Calculate intra cluster distance of chosen cluster: ---- QUALITY METRICS ---------
        ids_nonNoise = np.where(cluster_labels == chosen_cluster_ID)
        # intra_clust_dists, inter_clust_dists = clt_metrics.intra_cluster_distance(embs_total_faces,cluster_labels)
        # silh = sklearn.metrics.silhouette_samples(embs_total_faces,cluster_labels)
        # avg_intra_clt_dist = intra_clust_dists[ids_nonNoise].mean()
        # avg_inter_clt_dist = inter_clust_dists[ids_nonNoise].mean()
        # avg_silh = silh[ids_nonNoise].mean()
        # print("For ", participant, " intra_clt_dist: ", str(avg_intra_clt_dist), " , inter_clt_dist: ", str(avg_inter_clt_dist))
        # print("Shilouette: ", str(avg_silh))
        # print("Nimgs: ", str(len(ids_nonNoise[0])))
        # Save images (as in case x...) :
        rnd_num = random.randint(0, 1000)
        new_cluster_labels = -1*rnd_num*np.ones(shape=len(cluster_labels), dtype=int)
        new_cluster_labels[ids_nonNoise] = clt_index
        final_dict_assignation = {participant: [clt_index],
                                  "ZZUNK":[rnd_num]}

        cltAssignation.save_faces_and_embeddings_from_clusters(output_path_imgs, embs_total_faces,
                                                               embs_total_labels,
                                                               bbox_total_faces, [clt_index,rnd_num],
                                                               final_dict_assignation, new_cluster_labels)
        #Save images of complete cluster per participant
        if(debug):
            output_path_imgs_debug = os.path.join(output_path_imgs, "ZZdebug", participant)

            final_dict_assignation_debug = {participant: [chosen_cluster_ID]}
            os.makedirs(output_path_imgs_debug, exist_ok=True)
            cltAssignation.save_faces_and_embeddings_from_clusters(output_path_imgs_debug,embs_total_faces,embs_total_labels,
                                                                   bbox_total_faces,labels_ID, final_dict_assignation_debug,clt.labels_)

    def consolidate_clusters_numImgs(self,participant, quality_metric, input_path_models, input_path_parameters):
        """

        :param participant:
        :param quality_metric:
        :param input_path_models:
        :param input_path_parameters:
        :return:
        """
        #Load best parameters per participant
        chosen_cluster_ID = -1
        clt = None
        input_path_parameters_participant = os.path.join(input_path_parameters, participant)
        input_path_models_participant = os.path.join(input_path_models, participant)
        try:
            first_param,snd_param = loader.load_best_parameters_new(input_path_parameters_participant, quality_metric)
            #Load cluster
            path_cluster = os.path.join(input_path_models_participant,"cluster_"+self.first_param_name+"_"+str(first_param)+"_"+
                                        self.second_param_name+"_"+str(snd_param)+".sav")
            clt = loader.load_cluster(path_cluster)
            labels_ID = np.unique(clt.labels_)
            num_unique_faces = len(np.where(labels_ID > -1)[0])
            non_noise_data_index = np.where(clt.labels_!=-1)
            #Choose predominant cluster
            if(num_unique_faces>1):
                df = pd.DataFrame(clt.labels_[non_noise_data_index], columns=['cluster_ID'])
                df["n_labels"] = 1
                df_count = df.groupby('cluster_ID').sum()
                chosen_cluster_ID = df_count["n_labels"].argmax()
            elif(num_unique_faces==1):
                chosen_cluster_ID = list(set(labels_ID).difference([-1]))[0]
            else:
                return chosen_cluster_ID, clt
        except FileNotFoundError:
            pass
        return chosen_cluster_ID, clt

    def create_clusters_minDist(self, participant, quality_metric, input_path_models_current_DS, input_path_parameters_current_DS, input_path_embs_current_DS,
                                input_path_parameters_DS_2_compare,input_path_embsSum_DS_2_compare, k=-1):
        chosen_cluster_ID = -1
        clt_curr = None
        print("Processing: ", participant)
        if(participant=="Rocio Orellana"):
            print("debub")
        #Load current non_trust list:
        trust_ids_path_2compare = os.path.join(input_path_parameters_DS_2_compare,"non_trust_IDs.csv")
        df_trust = pd.read_csv(trust_ids_path_2compare, sep=";", header=0)
        row_participant = df_trust.loc[df_trust['ID_name'] == participant]
        good_cluster_candidate = False
        if(len(row_participant)>=1 and not row_participant["toCheck"].values[0] and row_participant["indiv_cluster_quality"].values[0]>-1):
            good_cluster_candidate = True
        try:
            first_param_curr,snd_param_curr = loader.load_best_parameters_new(os.path.join(input_path_parameters_current_DS, participant), quality_metric)
            #Load cluster
            path_cluster_curr = os.path.join(input_path_models_current_DS,participant,"cluster_"+self.first_param_name+"_"+str(first_param_curr)+"_"+
                                        self.second_param_name+"_"+str(snd_param_curr)+".sav")
            clt_curr = loader.load_cluster(path_cluster_curr)
            labels_ID_curr = np.unique(clt_curr.labels_)
            num_unique_faces_curr = len(np.where(labels_ID_curr > -1)[0])
            if(good_cluster_candidate and num_unique_faces_curr>1):
                #Load data of cluster to compare:
                key_name = ''
                for embs_zip_name in os.listdir(input_path_embsSum_DS_2_compare):
                    if(participant.replace(" ","_") in embs_zip_name or participant.replace("_", " ") in embs_zip_name):
                        key_name = embs_zip_name
                path_embs = os.path.join(input_path_embsSum_DS_2_compare, key_name)
                data = np.load(path_embs, allow_pickle=True)
                embs_2compare, _ = data['arr_0'], data['arr_1']
                #Extract closest cluster
                #Load embs current cluster
                path_embs_curr = os.path.join(input_path_embs_current_DS, participant+".npz")
                data_curr = np.load(path_embs_curr, allow_pickle=True)
                embs_curr, _ = data_curr['arr_0'], data_curr['arr_1']
                min_dist = sys.maxsize
                #remove -1
                new_list_labels_IDs = list(labels_ID_curr)
                new_list_labels_IDs.remove(-1)
                for clt_label in new_list_labels_IDs:
                    index_current_cluster = np.where(clt_curr.labels_==clt_label)
                    curr_embs_4_label = embs_curr[index_current_cluster]
                    sum_min_dist = self.calculate_min_dist(curr_embs_4_label, embs_curr, k=k)
                    if(sum_min_dist<min_dist):
                        min_dist = sum_min_dist
                        chosen_cluster_ID = clt_label
            else:
                input_path_models_current_DS = os.path.join(input_path_models_current_DS,participant)
                input_path_parameters_current_DS = os.path.join(input_path_parameters_current_DS,participant)
                chosen_cluster_ID, clt_curr = self.consolidate_clusters_numImgs(quality_metric, input_path_models_current_DS, input_path_parameters_current_DS)
        except FileNotFoundError:
            pass
        return chosen_cluster_ID, clt_curr

    def calculate_min_dist(self, embs_cluster_curr, embs_cluster_comp, k = 10):
        matrix_ref_label = euclidean_distances(embs_cluster_curr, embs_cluster_comp)
        added_points = []
        #Get minimum distance k points:
        for i in range((matrix_ref_label.shape[0])):
            min_val = np.min(matrix_ref_label[i, :])
            added_points.append(min_val)
        added_points = sorted(added_points)
        if(k>0):
            avg_dist = np.sum(added_points[0:k])/k
        else:
            avg_dist = np.mean(added_points)
        return avg_dist



if __name__ == "__main__":
    #parse input arguments
    clustering_args_obj = IndividualClusterConsolidatorArgs()
    args = clustering_args_obj.parse()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    #Input/Output dir for loading/saving clustering results
    input_dir = os.path.join(args.output_dir,"Cfacial_clustering_results",args.dataset)
    new_output_dir = args.output_dir

    extension = os.listdir(args.program_participants_folder)[0].split(".")[-1] \
        if ("." in os.listdir(args.program_participants_folder)[0]) else ''
    #programs & participants per program
    programs = sorted(os.listdir(args.root_path_MTCNN_results)) if (
            args.program_name == None or args.program_name == 'None') else [args.program_name]
    participants_per_program = {program: list(set(['OTHER'] +
                               loader.load_list_of_tertulianos(
                               args.program_participants_folder, program,
                               extension)))
                                 for program in programs}
    #POPULARITY MODULE
    if (args.popularity_df_path != ""):
        print("INCLUDING POPULARITY MODULE")
        df_popularity = pd.read_csv(args.popularity_df_path,  sep=";", header=0)


    for program in programs:
        #output_path_imgs = os.path.join(new_output_dir, "Cfacial_clustering_results", args.dataset, "x_pruebas", program)
        #Create individual clusters per participant in program
        try:
            participants_per_program[program].remove("OTHER")
        except ValueError:
            pass
        #Create new folders
        output_path_MTCNN = os.path.join(new_output_dir, "BMTCNN_results", args.dataset, program)
        output_path_imgs = os.path.join(new_output_dir, "Cfacial_clustering_results", args.dataset, "IND",
                                        args.consolidation_type,"assignation_x",
                                        "clustering_th_0",
                                        "cluster_filtering_assignation", program,
                                        "x_" + args.first_param_name + "_0_min_samples_0")

        input_path_models = os.path.join(input_dir, "cluster_models", program)
        input_path_parameters = os.path.join(input_dir, "cluster_graphics", "parameters", program)
        for participant in sorted(participants_per_program[program]):
            clustering_obj = IndividualClusterConsolidator(args.root_path_MTCNN_results, program, new_output_dir,
                                         [participant], args.dataset,
                                         args.first_param_name)
            # Define OUTPUT directories
            input_path_mtcnn = os.path.join(args.root_path_MTCNN_results, program, "mtcnn_debug_sum")
            dict_participants_clusters = {}
            index_cluster = 0
            if(args.consolidation_type=="minDist"):
                #info of dataset to compare:
                input_path_parameters_DS_2_compare = os.path.join(args.reference_parameters_path, program)
                input_path_embs_DS_2_compare = os.path.join(args.reference_embs_path)
                chosen_cluster_ID, clt = clustering_obj.create_clusters_minDist(participant, args.quality_metric,
                                                                                input_path_models,
                                                                                input_path_parameters, os.path.join(args.root_path_MTCNN_results, program, "embeddings_sum"),
                                                                                input_path_parameters_DS_2_compare,
                                                                                input_path_embs_DS_2_compare, k=args.k)
            else:
                chosen_cluster_ID, clt = clustering_obj.consolidate_clusters_numImgs(participant, args.quality_metric,
                                                                                     input_path_models,
                                                                                     input_path_parameters)

            #POPULARITY MODULE:
            if (args.popularity_df_path!=""):
                is_popular = df_popularity.loc[df_popularity["id"]==participant, "popular"].values[0]
                if(not is_popular):
                    chosen_cluster_ID = -1


            if(chosen_cluster_ID!=-1):
                if(args.save_as_MTCNN):
                    #Save as MTCNN in order to apply a second clustering step
                    clustering_obj.save_MTCNN_from_individual_clusters(participant, chosen_cluster_ID, clt, output_path_MTCNN)
                else:
                    #Save as clusters
                    clustering_obj.save_clusters_from_individual_clusters(participant, chosen_cluster_ID, clt,
                                                                          output_path_imgs, debug=True, clt_index=0)



