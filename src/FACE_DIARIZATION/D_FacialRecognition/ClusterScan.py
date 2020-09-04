"""
    DBSCAN or HDBSCAN clustering. This class also performs a greedy search in 2 DBSCAN/HDBSCAN parameter assed as first-param-name
    and second-param-name
	author: Cristina Luna, Ricardo Kleinlein
	date: 05/2020
	Usage:
		python3 ClusterScan.py
		--root-path-MTCNN-results ~/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN
        --program-participants-folder ~/test/
        --first-param-name eps
        --first-param-lower 0.5
        --first-param-upper 12
        --first-param-step 0.5
        --metric euclidean
        --cluster-instance DBSCAN
        --second-param-value [10]
        --metric euclidean
        --dataset Google
        --output-dir ~/supervised_results/BASELINE_OFICIAL_LN24H_v2
        --individual-clusters
        --quality-metric silhouette
        --with-previous-individual-clusters
        --with-previous-common-clusters
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
		-h, --help	Display script additional help
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.utils.loader as loader
import src.utils.saver as saver
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import pickle
import src.metrics.metrics_clustering as metrics_clustering
from sklearn import metrics as sklearn_metrics
from src.BaseArgs import ClusteringArgs
import src.utils.files_utils as tools




class ClusterScan():

    def __init__(self, root_path_MTCNN_results, program_name, first_param_lower, first_param_upper, first_param_step,
                 second_parameter_list, output_dir, program_participants, metric="euclidean",
                 first_param_name = "eps", second_param_name ="min_samples",cluster_instance="DBSCAN"):
        #cluster parameters to scan:
        self.first_param_values = sorted(list(np.linspace(first_param_upper,first_param_lower,(first_param_upper-first_param_lower)/first_param_step, endpoint=False))+[float(first_param_lower)])
        self.second_param_values = second_parameter_list
        self.metric = metric
        self.cluster_instance = None
        self.first_param_name = first_param_name
        self.second_param_name = second_param_name
        if(cluster_instance=="HDBSCAN"):
            self.cluster_instance = HDBSCAN()
        else:
            self.cluster_instance = DBSCAN()
        #Input
        self.root_input_path = root_path_MTCNN_results
        self.program = program_name
        self.program_participants = program_participants
        #Output
        self.output_dir = output_dir
        #Best cluster data
        self.best_cluster_quality = -1


    def load_cluster(self, path_cluster_model,input_embs, cluster_parameters={}):
        """
        Load DBSCAN model from path_cluster_model. If the cluster does not exists, then train it.
        :param path_cluster_model: Path to load/save cluster
        :param input_embs: Input data to train cluster. In this case, faces embeddings
        :param cluster_parameters: Dict with the cluster parameters to load:
            e.g. for DBSCAN: {'eps': 1.5, 'min_samples':10,'metric':'euclidean'}
        :param min_sample: min_sample DBSCAN parameter
        :return:
         - clt: trained cluster (DBSCAN or HDBSCAN)
        """
        if(type(self.cluster_instance) is type(HDBSCAN())):
            if("min_cluster_size" in cluster_parameters):
                cluster_parameters["min_cluster_size"] = int(cluster_parameters["min_cluster_size"])
        try:
            if (os.path.exists(path_cluster_model)):
                clt = pickle.load(open(path_cluster_model, 'rb'))
            else:
                self.cluster_instance.set_params(**cluster_parameters)
                clt = self.cluster_instance
                clt.fit(input_embs)
                pickle.dump(clt, open(path_cluster_model, 'wb'))
        except ModuleNotFoundError:
            self.cluster_instance.set_params(**cluster_parameters)
            clt = self.cluster_instance
            clt.fit(input_embs)
            pickle.dump(clt, open(path_cluster_model, 'wb'))
        return clt

    def cluster_parameters_scan(self, output_path_models, output_path_graphics):
        """
           Scan parameters of DBSCAN in terms of min_samples and epsilon & save the generated clusters for a posterior
           analysis of the cluster that best fit
           to the expected number of clusters
               :param program: (str) : Name of the program to analyse
               :param output_path_models (str): Path where the clusters will be saved
               :param output_path_graphics (str): Path where the graphs and the other metadata will be saved

           """
        input_path_embs = os.path.join(self.root_input_path, self.program, "embeddings_sum")
        input_path_bbox = os.path.join(self.root_input_path, self.program, "boundingboxes_sum")
        os.makedirs(output_path_graphics, exist_ok=True)
        #Create list of trust participants based on mean faces in imgs downloaded

        # load embs & bboxes for participants of program
        bbox_total_faces, bbox_total_labels, embs_total_faces, embs_total_labels = loader.load_bboxes_and_embs(
            input_path_bbox, input_path_embs, self.program_participants)
        embs_total_labels2 = [x.split("/")[0] for x in embs_total_labels]
        print('\n' + str(len(self.program_participants)) + ' PARTICIPANTS in ' + self.program)
        # If there are images for person, then create clustering
        if(len(bbox_total_labels)>1):
            os.makedirs(output_path_models, exist_ok=True)

            # Scan eps and min_sample values
            df_results = pd.DataFrame()
            cluster_matrix = np.zeros(shape=(len(self.first_param_values), len(self.second_param_values)))  # number of clusters created by DBSCAN
            avg_silhouette_matrix = np.zeros(
                shape=(len(self.first_param_values), len(self.second_param_values)))  # value of avg_silhouette for each combination in the scan
            calinshi_harabasz_matrix = np.zeros(shape=(
            len(self.first_param_values), len(self.second_param_values)))  # value of calinshi_harabasz score for each combination in the scan
            v_measure_matrix = np.zeros(shape=(len(self.first_param_values), len(self.second_param_values)))
            i, j = 0, 0
            fig = plt.figure()
            for snd_param_val in self.second_param_values:
                clusters = []
                i = 0
                for first_param_val in self.first_param_values:
                    print("\n[INFO] Clustering for "+self.first_param_name+" = " + str(first_param_val) + " and "+self.second_param_name+" = " + str(snd_param_val))
                    name_cluster = "cluster_"+self.first_param_name+"_" + str(first_param_val) + "_"+self.second_param_name+"_" + str(snd_param_val) + ".sav"
                    path_cluster_model = os.path.join(output_path_models, name_cluster)
                    cluster_params = {self.first_param_name:first_param_val,
                                      self.second_param_name: snd_param_val,
                                      "metric":self.metric}
                    clt = self.load_cluster(path_cluster_model, embs_total_faces, cluster_params)
                    numUniqueFaces = len(np.where(np.unique(clt.labels_) > -1)[0])
                    clusters.append(numUniqueFaces)
                    df_results = df_results.append(
                        pd.DataFrame([[first_param_val, snd_param_val, numUniqueFaces]], columns=[self.first_param_name, self.second_param_name, "n_clusters"]))
                    print("> Clusters: {}".format(numUniqueFaces))
                    cluster_matrix[i, j] = numUniqueFaces
                    try:
                        y_pred = clt.labels_
                        avg_silhouette_matrix[i, j] = silhouette_score(embs_total_faces, y_pred)
                        calinshi_harabasz_matrix[i, j] = calinski_harabasz_score(embs_total_faces,y_pred)
                        homogeneity, completeness, v_measure = sklearn_metrics.homogeneity_completeness_v_measure(
                            embs_total_labels2, y_pred)
                        v_measure_matrix[i, j] = v_measure
                    except ValueError:
                        avg_silhouette_matrix[i, j] = -1
                        calinshi_harabasz_matrix[i, j] = -1
                        v_measure_matrix[i, j] = -1
                    i += 1
                j += 1
                plt.plot(self.first_param_values, clusters, label=self.second_param_name+' = ' + str(snd_param_val))
            plt.xlabel(self.first_param_name)
            plt.ylabel('# Number of clusters')
            plt.legend(fontsize="small")
            plt.grid(True, which='both')
            plt.minorticks_on()
            # save data
            plt.savefig(os.path.join(output_path_graphics, 'cluster_graphic.png'))
            plt.close(fig)
            # Save graphics and matrixes generated for posterior choice of best cluster
            self.save_clustering_result_graphics(output_path_graphics, df_results, cluster_matrix, avg_silhouette_matrix,
                                            calinshi_harabasz_matrix, v_measure_matrix)


    def save_clustering_result_graphics(self, output_path_graphics,df_results,cluster_matrix,avg_silhouette_matrix,
                                        calinshi_harabasz_matrix, v_measure_matrix):
        """
        Save information about cluster training
        :param output_path_graphics: Path to save matrixes with metrics extracted from clusters
        :param df_results: Dataframe with columns: [eps,min_sample,number_clusters] that contain results of that clustering configuration
        :param cluster_matrix: Matrix with as many rows as epsilons tested and as many columns as min_samples. In each
        cell it will contain the total number of clusters per (eps,min_sample) combination
        :param avg_silhouette_matrix: Matrix with as many rows as epsilons tested and as many columns as min_samples. In each
        cell it will contain the results obtained of sklearn silhouette_score metric.
        :param calinshi_harabasz_matrix: Similar to previous but in this case the matrix contains the results of the calinski_harabasz_score
        :param v_measure_matrix: Similar to previous but in this case the matrix contains the results of the homogeneity_completeness_v_measure
        """
        out_path_csv = os.path.join(output_path_graphics, 'combination_parameters_Ncluster.csv')
        df_results.to_csv(out_path_csv, index=False)
        dict_matrixes = {'cluster_matrix':cluster_matrix,
                         'silhouette_matrix':avg_silhouette_matrix,
                         'calinski_harabasz_matrix':calinshi_harabasz_matrix,
                         'v_meassure_matrix':v_measure_matrix}
        for matrix_name in dict_matrixes.keys():
            #Generate csv from matrixed for silhouette, calinshi, v_measure and n_clusters
            (pd.DataFrame(dict_matrixes[matrix_name])).to_csv(
                os.path.join(output_path_graphics, matrix_name+'.csv'), index=False, header=False)
            # Generate heatmap from matrixes for silhouette, calinshi, v_measure and n_clusters
            saver.save_heatmap_from_matrix(os.path.join(output_path_graphics, matrix_name+'.csv'),
                                            os.path.join(output_path_graphics, matrix_name+".png"),
                                            self.first_param_values,self.second_param_values,format=".4f", figsize=(45, 10))


    def save_best_cluster_parameters(self, output_path_graphics, quality_metric="silhouette", use_global_max=True):
        """
        Save the best cluster combination from
        :param output_path_graphics:
        :param quality_metric:
        """
        #Load matrix:
        if(os.path.exists(os.path.join(output_path_graphics, quality_metric+"_matrix.csv"))):
            df_metrics = pd.read_csv(os.path.join(output_path_graphics, quality_metric+"_matrix.csv"), header=None)
            #Get best parameters
            #Global max
            if(use_global_max):
                row_index,col_index = np.where(df_metrics.values == np.max(df_metrics.values))
            else: #local
                #First local max (1st row)
                local_max = tools.get_local_max(df_metrics.values)[0]
                row_index, col_index = np.where(df_metrics.values == local_max)
            # Save best parameters
            np.savez_compressed(os.path.join(output_path_graphics, quality_metric+"_best_cluster_parameters.npz"),
                                [(self.first_param_values[row_index[0]], self.second_param_values[col_index[0]])])
            self.best_cluster_quality = np.max(df_metrics.values)





if __name__ == "__main__":
    #parse input arguments
    clustering_args_obj = ClusteringArgs()
    args = clustering_args_obj.parse()

    #Output dir for saving clustering results
    new_output_dir = os.path.join(args.output_dir,"Cfacial_clustering_results",args.dataset)
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
    #Extract clusters per program
    for program in programs:
        input_path_mtcnn = os.path.join(args.root_path_MTCNN_results, program, "mtcnn_debug_sum")
        if(args.individual_clusters):
            #Create individual clusters per participant in program
            try:
                participants_per_program[program].remove("OTHER")
            except ValueError:
                pass

            for participant in sorted(participants_per_program[program]):
                clustering_obj = ClusterScan(args.root_path_MTCNN_results, program, args.first_param_lower,
                                             args.first_param_upper,
                                             args.first_param_step, args.second_param_value, new_output_dir,
                                             [participant],
                                             args.metric, args.first_param_name, cluster_instance=args.cluster_instance)
                # Define OUTPUT directories
                output_path_models = os.path.join(new_output_dir, "cluster_models", program, participant)
                output_path_graphics = os.path.join(new_output_dir, "cluster_graphics", "parameters", program, participant)
                clustering_obj.cluster_parameters_scan(output_path_models, output_path_graphics)
                clustering_obj.save_best_cluster_parameters(output_path_graphics, args.quality_metric, use_global_max=args.silhouette_global_max)
                #clustering_obj.check_trust_of_imgs(input_path_mtcnn, output_path_graphics)
                clustering_obj = None
            #Save complete csv with good/bad clusters:
            # df_trust_complete = pd.DataFrame([], columns=["ID_name", "mean_faces_per_ID", "toCheck", "indiv_cluster_quality", "has_cluster"])
            # for participant in sorted(participants_per_program[program]):
            #     path_graph = os.path.join(new_output_dir, "cluster_graphics", "parameters", program, participant, "non_trust_IDs.csv")
            #     df_trust_complete = df_trust_complete.append(pd.read_csv(path_graph, header=0, sep=";"))
            # df_trust_complete.to_csv(os.path.join(new_output_dir, "cluster_graphics", "parameters", program,"non_trust_IDs.csv"), header=True, sep=";", index=False)

        else:
            # Create complete cluster per program
            folder_2_save_data = ""
            if (args.with_previous_common_clusters):
                folder_2_save_data += "COMMON"
            if (args.with_previous_individual_clusters):
                if ("COMMON" in folder_2_save_data):
                    folder_2_save_data = "IND_" + folder_2_save_data
                else:
                    folder_2_save_data = "IND"

            clustering_obj = ClusterScan(args.root_path_MTCNN_results, program, args.first_param_lower,
                                         args.first_param_upper,
                                         args.first_param_step, args.second_param_value, new_output_dir,
                                         participants_per_program[program],
                                         args.metric, args.first_param_name, cluster_instance=args.cluster_instance)
            # Define OUTPUT directories
            output_path_models = os.path.join(new_output_dir, "cluster_models", program, folder_2_save_data)
            output_path_graphics = os.path.join(new_output_dir, "cluster_graphics", "parameters", program, folder_2_save_data)
            clustering_obj.cluster_parameters_scan(output_path_models, output_path_graphics)
            clustering_obj.save_best_cluster_parameters(output_path_graphics, args.quality_metric, use_global_max=args.silhouette_global_max)
            clustering_obj.best_cluster_quality = -1
            #clustering_obj.check_trust_of_imgs(input_path_mtcnn, output_path_graphics)
            clustering_obj = None
