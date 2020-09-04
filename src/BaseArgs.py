"""
    author: Ricardo Kleinlein, Cristina Luna
    date: 06/2020
    Script to define the Arguments class. Every script will have its own
    set of arguments as a rule, though some may be shared between tasks.
    These objects are not thought to be used independently, but simply
    as a method to automate the argument parsing between scripts in the
    retrieval pipeline.
"""

import os
import argparse
import __main__ as main
from ast import literal_eval

upfold = lambda s: os.path.dirname(s)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BaseArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--output-dir",
            type=str,
            default="results",
            help="Directory to export the script s output to")

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()
        self.correct()

        if not self.args.quiet:
            print('-' * 10 + ' Arguments ' + '-' * 10)
            print('>>> Script: %s' % (os.path.basename(main.__file__)))
            print_args = vars(self.args)
            for key, val in sorted(print_args.items()):
                print('%s: %s' % (str(key), str(val)))
            print('-' * 30)
        return self.args

    def correct(self):
        """Assert ranges of params, mistypes..."""
        raise NotImplementedError

class DSPreparatorArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseArgs.initialize(self)
        self.parser.add_argument(
            "--input-path-labels-rttm",
            type=str,
            default=None,
            help="Path to folder with program labels")
        self.parser.add_argument(
            "--output-path-labels-rttm",
            type=str,
            default=None,
            help="Path to folder with program labels with new format")
        self.parser.add_argument(
            "--output-path-participants-folder",
            type=str,
            default=None,
            help="Path to folder with names of participants")
        self.parser.add_argument(
            "--category",
            type=str,
            default=None,
            help="Category of rttms (FACEREF, SPEAKER,SEASON...)")
        self.parser.add_argument(
            "--programs-folder",
            type=str,
            default=None,
            help="Folder with the videos/programs for extracting the names")

        self.parser.add_argument(
            "--semi-supervised",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we have an rttm labels with participant names, "
                 "else False (in this case we should have participants in a file as a list) "
                 "False e.g: "
                 "Juan Rojas"
                 "Pepito López"
                 "....")
        self.parser.add_argument(
            "--quiet",
            type=bool,
            default=True,
            help="")

    def correct(self):
        assert os.path.isdir(self.args.input_path_labels_rttm)
        assert isinstance(self.args.output_path_labels_rttm,str)
        assert isinstance(self.args.output_path_participants_folder,str)
        assert isinstance(self.args.category, str) and self.args.category in ["FACE","FACEREF", "SPKREF"]
        assert (os.path.isdir(self.args.programs_folder))
        assert (isinstance(self.args.semi_supervised,bool))



class QueryArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseArgs.initialize(self)
        self.parser.add_argument(
            "--participants-path",
            type=str,
            default=None,
            help="Path to the files with the names of the participants "
                 "(See DSPreparator.py outputs ")

        self.parser.add_argument(
            "--imgs-2-download",
            type=int,
            default=150,
            help="Max. number of imgs to download per participant in participants_path [default: 150]")

        self.parser.add_argument(
            "--chrome-driver-path",
            type=str,
            default=None,
            help="Path where chrome_driver was downloaded (e.g. ~/chromedriver_linux64/chromedriver)")


        self.parser.add_argument(
            "--logs-path",
            type=str,
            default="",
            help="Directory to save the logs info about the URL, path... of the downloaded imgs [default: output-dir/logs]")

        self.parser.add_argument(
            "--extra-info",
            type=str,
            default='',
            help="Keywords to append after participant name before downloading [default: Not append anything]")

        self.parser.add_argument(
            "--quiet",
            type=bool,
            default=True,
            help="")

    def correct(self):
        assert os.path.isfile(self.args.participants_path)
        assert isinstance(self.args.imgs_2_download, int)
        assert os.path.isfile(self.args.chrome_driver_path)
        assert isinstance(self.args.output_dir, str)
        assert isinstance(self.args.logs_path, str)
        assert isinstance(self.args.extra_info, str)


class BingQueryArgs(QueryArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        QueryArgs.initialize(self)
        self.parser.add_argument(
            "--ultralytics-repo-path",
            type=str,
            default="../../../Alibs/google_images_download_bing",
            help="Path to the ultralytics downloaded repository"
                 "(download web: https://github.com/ultralytics/google-images-download) ")

    def correct(self):
        QueryArgs.correct(self)
        assert os.path.isdir(self.args.ultralytics_repo_path)



class FrameExtractorArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseArgs.initialize(self)
        self.parser.add_argument(
            "--input-videos-folder",
            type=str,
            default=None,
            help="Path to program's video folder (folder with the video/s to extract frames)")
        self.parser.add_argument(
            "--video-name",
            type=str,
            default=None,
            help="Name of the video to analyse (if we want to extract a single video, else None -> "
                 "If None,then analyse all videos in input_videos_folder)")
        self.parser.add_argument(
            "--fps",
            type=int,
            default=0,
            help="Frames per second to extract [default:0 - Extract the default fps of the program video]")
        self.parser.add_argument(
            "--frame-height",
            type=int,
            default=0,
            help="Height of the frames [default: heigth of the video]")
        self.parser.add_argument(
            "--frame-width",
            type=int,
            default=0,
            help="Width of the frames [default: width of the video]")
        self.parser.add_argument(
            "--parallel-processing",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we want to use parallel processing to extract frames of all the videos in input_videos_folder")
        self.parser.add_argument(
            "--extraction-lib",
            type=str,
            default="ffmpeg",
            help="Extract frames with opencv or with ffmpeg [default: ffmpeg] (Options: ffmpeg or opencv)")
        self.parser.add_argument(
            "--quiet",
            type=bool,
            default=True,
            help="")

    def correct(self):
        assert os.path.isdir(self.args.input_videos_folder)
        assert isinstance(self.args.output_dir, str)
        assert (isinstance(self.args.video_name, str) or self.args.video_name == None)
        assert isinstance(self.args.fps, int)
        assert isinstance(self.args.frame_height, int)
        assert isinstance(self.args.frame_width, int)
        assert isinstance(self.args.parallel_processing, bool)
        assert isinstance(self.args.extraction_lib, str) and (
                    self.args.extraction_lib == "opencv" or self.args.extraction_lib == "ffmpeg")


class FaceDetEncArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseArgs.initialize(self)
        self.parser.add_argument(
            "--root-input-folder",
            type=str,
            default=None,
            help="Path to the root directory that has programs as subdirectories with their frames")
        self.parser.add_argument(
            "--program-name",
            type=str,
            default=None,
            help="Name of the program to analyse (if we want to process a single program, else None & all the programs "
                 "will be processed) [default: None]")
        self.parser.add_argument(
            "--encoding-model",
            type=str,
            default='../../../Bresources/pre_trained_models/face_embs_model/facenet_keras.h5',
            help="Path to face encoding model weights [default: Keras Facenet")
        self.parser.add_argument(
            "--face-detector",
            type=str,
            default='MTCNN',
            help="Model to use MTCNN [1] or HaarCascade [2]")
        self.parser.add_argument(
            "--program-participants-folder",
            type=str,
            default=None,
            help="Path to the files with the names of the participants "
                 "(See DSPreparator.py outputs ")
        self.parser.add_argument(
            "--imgs-2-maintain",
            type=int,
            default=0,
            help="Number of images per participant/program to maintain. If 0, then maintain all the images[default: 0]")

        self.parser.add_argument(
            "--face-threshold",
            type=float,
            default=0.0,
            help="Probability[0-1] of accept a face as correct . Those faces below the face_threshold will not be considered in what follows."
                 "[default: 0.0]")

        self.parser.add_argument(
            "--extract-single-face",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we want to extract a single face from frame (This face will be the biggest one in the image)[default: False]")
        self.parser.add_argument(
            "--quiet",
            type=bool,
            default=True,
            help="")

    def correct(self):
        assert os.path.isdir(self.args.root_input_folder)
        assert (isinstance(self.args.program_name, str) or self.args.program_name == None)
        assert (os.path.isfile(self.args.encoding_model))
        assert isinstance(self.args.face_detector, str) and (self.args.face_detector in ["MTCNN","HaarCascade"])
        assert isinstance(self.args.output_dir, str)
        assert (os.path.isdir(self.args.program_participants_folder))
        assert isinstance(self.args.imgs_2_maintain, int)
        assert isinstance(self.args.face_threshold, float) and\
               (self.args.face_threshold <= 1 and self.args.face_threshold >= 0)
        assert isinstance(self.args.extract_single_face, bool)


class RefactorDSArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseArgs.initialize(self)
        self.parser.add_argument(
            "--root-input-path",
            type=str,
            default=None,
            help="Path to the root folder whose sub-folders need to be refactored/renamed")
        self.parser.add_argument(
            "--set-to-analyse",
            type=str,
            default=None,
            help="Refactor frames structure in order to be in concordance with OCR structure: "
                 "program0/"
                 "  |_participant0/"
                 "      |_participant0_frame0.png"
                 "      |_participant0_frame1.png"
                 "      |...                     "
                 "  |_participant1/"
                 "  ..."
                 "In PROGRAM there is one unique participant with label OTHER"
                 "Options: (Google, PROGRAM)")

        self.parser.add_argument(
            "--program-participants-folder",
            type=str,
            default=None,
            help="Path to the files with the names of the participants "
                 "(See DSPreparator.py outputs ")
        self.parser.add_argument(
            "--quiet",
            type=bool,
            default=True,
            help="")

    def correct(self):
        assert os.path.isdir(self.args.root_input_path)
        assert isinstance(self.args.set_to_analyse, str) and (
                self.args.set_to_analyse == "Google" or self.args.set_to_analyse == "OCR" or
                self.args.set_to_analyse == "PROGRAM")
        assert (os.path.isdir(self.args.program_participants_folder))



class ClusteringArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseArgs.initialize(self)

        self.parser.add_argument(
            "--root-path-MTCNN-results",
            type=str,
            default=None,
            help="Root path to the results obained by MTCNN (embs, bbox, mtcnn_debub, bbox_summ)")
        self.parser.add_argument(
            "--program-participants-folder",
            type=str,
            default=None,
            help="Path to the files with the names of the participants "
                 "(See DSPreparator.py outputs ")
        self.parser.add_argument(
            "--program-name",
            type=str,
            default=None,
            help="Name of the program to analyse (if we want to process a single program, else None)")
        self.parser.add_argument(
            "--first-param-name",
            type=str,
            default='eps',
            help="Name of the first param as it appears in the DBSCAN and HDBSCAN libraries."
                 "DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"
                 "HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan "
                 "[default: eps (DBSCAN param.)]")
        self.parser.add_argument(
            "--first-param-lower",
            type=float,
            default=0.5,
            help="Lower value for first parameter [default: 0.5]")
        self.parser.add_argument(
            "--first-param-upper",
            type=float,
            default=12.,
            help="Upper value for fisrt parameter [default: 12]")
        self.parser.add_argument(
            "--first-param-step",
            type=float,
            default=0.5,
            help="Step between 2 consecutive values of the first parameter")

        self.parser.add_argument(
            "--second-param-name",
            type=str,
            default='min_samples',
            help="Name of the second param as it appears in the DBSCAN and HDBSCAN libraries."
                 "DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"
                 "HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan "
                 "[default: min_samples (DBSCAN & HBSCAN param.)]")
        self.parser.add_argument(
            "--second-param-value",
            type=str,
            default="[10]",
            help="List with second param values [default: [10]]")
        self.parser.add_argument(
            "--metric",
            type=str,
            default='euclidean',
            help="Distance metric in DBSCAN")
        self.parser.add_argument(
            "--dataset",
            type=str,
            default='',
            help="Dataset to extract clustering (e.g Google, OCR, Google_OCR, Google_Program ...")
        self.parser.add_argument(
            "--cluster-instance",
            type=str,
            default='DBSCAN',
            help = "Type of cluster to use. Options: DBSCAN or HDBSCAN")
        self.parser.add_argument(
            "--individual-clusters",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we want a cluster per participant,"
                 "else it creates a common cluster for all participants")
        self.parser.add_argument(
            "--quality-metric",
            type=str,
            default='silhouette',
            help="Metric to use as quality and select the best cluster combination of parameters."
                 " Options ( cluster, v_meassure, silhouette, calinski_harabasz)."
                 "cluster -> Select those parameters that obtain the max. number of clusters"
                 "v_meassure -> Select those parameters that obtain the max. v_meassure (See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html)"
                 "silhouette -> Select those parameters that obtain the max. silhouette (See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)")
        self.parser.add_argument(
            "--with-previous-individual-clusters",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we did a previous individual cluster per participant"
                 "else False. "
                 "This parameter is just informative in order to save results in different folders")

        self.parser.add_argument(
            "--with-previous-common-clusters",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we did a previous common cluster with all the participants"
                 "else False. "
                 "This parameter is just informative in order to save results in different folders")

        self.parser.add_argument(
            "--silhouette-global-max",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we want to choose the best cluster parameters combination based on global max. of the silhouette,"
                 " or (is False) based on the 1st local maximum of the silhouette")

        self.parser.add_argument(
            "--quiet",
            type=bool,
            default=True,
            help="")

    def correct(self):
        assert os.path.isdir(self.args.root_path_MTCNN_results)
        assert (os.path.isdir(self.args.program_participants_folder))
        # self.args.output_dir = os.path.join(
        #     os.path.dirname(self.args.program_csv), 'dbscan')
        assert isinstance(self.args.first_param_name, str)
        assert isinstance(self.args.first_param_lower, float) or isinstance(self.args.first_param_lower, int)
        assert isinstance(self.args.first_param_upper, float) or isinstance(self.args.first_param_upper, int)
        assert isinstance(self.args.first_param_step, float) or isinstance(self.args.first_param_step, int)
        assert isinstance(self.args.second_param_value, str)
        self.args.second_param_value = (self.args.second_param_value).strip().replace("[", "").replace("]", "").split(",")
        #Convert to integer
        self.args.second_param_value = [int(min_sample) for min_sample in self.args.second_param_value]
        assert isinstance(self.args.cluster_instance, str) and (self.args.cluster_instance in ["DBSCAN", "HDBSCAN"])
        assert isinstance(self.args.metric, str)
        dists = ['euclidean', 'cosine', 'manhattan', 'l1', 'l2']
        if self.args.metric not in dists:
            raise NotImplementedError('Pairwise distance not implemented')
        assert isinstance(self.args.dataset, str)
        assert isinstance(self.args.quality_metric, str)
        if self.args.quality_metric not in ["cluster", "v_meassure", "silhouette", "calinski_harabasz"]:
            raise NotImplementedError('Quality metric not implemented')
        assert isinstance(self.args.with_previous_individual_clusters, bool)
        assert isinstance(self.args.with_previous_common_clusters, bool)



class IndividualClusterConsolidatorArgs(ClusteringArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        ClusteringArgs.initialize(self)

        self.parser.add_argument(
            "--save-as-MTCNN",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we want save data as MTCNN output,"
                 "else it assignate clusters to participants based on num. of images per cluster"
                 "or by comparison with 'reference DS'")
        self.parser.add_argument(
            "--consolidation-type",
            type=str,
            default='numImgs',
            help="How to consolidate individual clusters and select the chosen cluster. (Options: numImgs, minDist)"
                 "If numImgs -> Assign the cluster with the max. number of images as the best, and thus assign it to the participant"
                 "If minDist -> Assign the similarest cluster to the participant based on a second datset ('reference DS')")
        self.parser.add_argument(
            "--k",
            type=int,
            default=-1,
            help="If consolidation-type=minDist is selected, it requires the k neighbours to compare and decide which is the best cluster")
        self.parser.add_argument(
            "--reference-parameters-path",
            type=str,
            default="",
            help="If consolidation-type=minDist is selected, it requires the path where to find the -csv with the trust/non-trust clusters."
                 "(default e.g. of path: args.output_dir/Cfacial_clustering_results/reference_dataset/cluster_graphics/parameters)")
        self.parser.add_argument(
            "--reference-embs-path",
            type=str,
            default="",
            help="If consolidation-type=minDist is selected, it requires the path where to find the clusters of the datset to compare as well as their embeddings."
                 "(default e.g. of path: args.output_dir/Cfacial_clustering_results/reference_dataset/x/assignation_x/clustering_th_x/cluster_filtering_assignation/PROGRAM/x_eps_x_min_samples_x/embeddings_sum)")

        self.parser.add_argument(
            "--random-seed",
            type=int,
            default=42,
            help="Random seed for generating splits of train/test [default: 42] ")

        self.parser.add_argument(
            "--popularity-df-path",
            type=str,
            default="",
            help="Path to the popularity dataframe if we want to use popularity, else not fill it [default: ""] ")


    def correct(self):
        ClusteringArgs.correct(self)
        assert isinstance(self.args.save_as_MTCNN, bool)
        assert isinstance(self.args.consolidation_type, str) and self.args.consolidation_type in ["numImgs", "minDist"]
        assert isinstance(self.args.k, int)
        if (self.args.consolidation_type == "minDist"):
            assert isinstance(self.args.reference_parameters_path, str) and os.path.exists(self.args.reference_parameters_path)
            assert isinstance(self.args.reference_embs_path, str) and os.path.exists(self.args.reference_embs_path)
        assert isinstance(self.args.random_seed, int)




# RESULTS ------------------------------------------------------------
class ResultsCommonArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseArgs.initialize(self)

        self.parser.add_argument(
            "--assignation-type",
            type=str,
            default="One2One",
            help="Type of cluster-participant assignation [Options: One2One or Many2One]."
                 "In an assingation One2One: One cluster (the best) may be assigned to One participant."
                 "In an Many2One assignation: Many clusters (those that accomplish the cuality criterion:qualityTh-valid-cluster)"
                 " may be assigned to One participant")

        self.parser.add_argument(
            "--qualityTh-valid-cluster",
            type=float,
            default=0.5,
            help="Percentage of images in cluster for being considered as good quality cluster for assigning it"
                 " to a participant [default: 0.5]")

        self.parser.add_argument(
            "--assignation-method",
            type=str,
            default="V_M",
            help="Method to use for assgination of clusters to participants[default: V_M (unique implmented by now)]")

        self.parser.add_argument(
            "--program-name",
            type=str,
            default=None,
            help="Name of the program to analyse (if we want to process a single program, else None)")

        self.parser.add_argument(
            "--dataset",
            type=str,
            default='',
            help="Dataset to extract clustering (e.g Google, OCR, Google_OCR, Google_Program ...")

        self.parser.add_argument(
            "--program-participants-folder",
            type=str,
            default=None,
            help="Path to the files with the names of the participants "
                 "(See DSPreparator.py outputs ")

        self.parser.add_argument(
            "--add-noise-class",
            type=str2bool, nargs='?',
            const=True,
            default=False,
            help="True if our model will include a noise class label, else False. If True, path to noise_data must be introduced"
                 "in noise_data_path (when it is a required parameter)")

        self.parser.add_argument(
            "--first-param-name",
            type=str,
            default='eps',
            help="Name of the first param as it appears in the DBSCAN and HDBSCAN libraries."
                 "DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"
                 "HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan "
                 "[default: eps (DBSCAN param.)]")

        self.parser.add_argument(
            "--second-param-name",
            type=str,
            default='min_samples',
            help="Name of the second param as it appears in the DBSCAN and HDBSCAN libraries."
                 "DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"
                 "HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan "
                 "[default: min_samples (DBSCAN & HBSCAN param.)]")

        self.parser.add_argument(
            "--with-previous-individual-clusters",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we did a previous individual cluster per participant"
                 "else False. "
                 "This parameter is just informative in order to save results in different folders")

        self.parser.add_argument(
            "--with-previous-common-clusters",
            type=str2bool, nargs='?',
            const=True, default=False,
            help="True if we did a previous common cluster with all the participants"
                 "else False. "
                 "This parameter is just informative in order to save results in different folders")

        self.parser.add_argument(
            "--quality-metric",
            type=str,
            default='silhouette',
            help="Metric to use as quality and select the best cluster combination of parameters."
                 " Options ( cluster, v_meassure, silhouette, calinski_harabasz)."
                 "cluster -> Select those parameters that obtain the max. number of clusters"
                 "v_meassure -> Select those parameters that obtain the max. v_meassure (See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html)"
                 "silhouette -> Select those parameters that obtain the max. silhouette (See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)")

    def correct(self):
        assignation_type = ['One2One', 'Many2One', 'x']
        if self.args.assignation_type not in assignation_type:
            raise NotImplementedError('Assignation type not implemented, try: One2One or Many2One')
        assert isinstance(self.args.qualityTh_valid_cluster, float) and self.args.qualityTh_valid_cluster <= 1
        if self.args.assignation_method not in ["V_M", "x", 'numImgs', 'minDist']:
            raise NotImplementedError('Assignation method '+self.args.assignation_method+'not implemented, try: V_M')
        assert (isinstance(self.args.program_name, str) or self.args.program_name == None)
        assert isinstance(self.args.dataset, str)
        assert (os.path.isdir(self.args.program_participants_folder))
        assert isinstance(self.args.add_noise_class, bool)
        assert isinstance(self.args.first_param_name, str)
        assert isinstance(self.args.second_param_name, str)
        if self.args.quality_metric not in ["cluster", "v_meassure", "silhouette", "calinski_harabasz"]:
            raise NotImplementedError('Quality metric not implemented')
        assert isinstance(self.args.with_previous_individual_clusters, bool)
        assert isinstance(self.args.with_previous_common_clusters, bool)


class ClusteringCleanerArgs(ResultsCommonArgs):
    def __init__(self):
        super().__init__()

    def initialize(self):
        ResultsCommonArgs.initialize(self)
        self.parser.add_argument(
            "--root-input-dir",
            type=str,
            default=None,
            help="Input dir where clustering results were saved. If None, then output dir is taken as reference and the data"
                 "will be search in default locations : output_dir/Cfacial_clustering_results/..")

        self.parser.add_argument(
            "--root-path-MTCNN-results",
            type=str,
            default=None,
            help="Root path to the results obained by MTCNN (embs, bbox, mtcnn_debub, bbox_summ)")

        self.parser.add_argument(
            "--quiet",
            type=bool,
            default=True,
            help="")

    def correct(self):
        ResultsCommonArgs.correct(self)
        assert (os.path.isdir(
            self.args.root_input_dir)) or self.args.root_input_dir == None or self.args.root_input_dir == 'None'
