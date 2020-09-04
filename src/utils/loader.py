import os
import pandas as pd
import numpy as np
import src.utils.files_utils as file_utils
import pickle
from keras.preprocessing import image


def load_list_of_tertulianos(path_tertuliano_file, program, extension=".csv"):
    path_tertuliano_ids = os.path.join(path_tertuliano_file, program+extension)
    tertulianos_df = pd.read_csv(path_tertuliano_ids, delimiter="@", header=None)
    tertulianos_list = list(tertulianos_df[0])
    for index in range(len(tertulianos_list)):
        value = tertulianos_list[index]
        if (value[-1] == " "):
            new_value = value[0:-1]
        else:
            new_value = value
        tertulianos_list[index] = file_utils.strip_accents(new_value)
    tertulianos_list = sorted(list(set(tertulianos_list)))
    return tertulianos_list


def get_participants(participants_path):
    """Get the name & twitter id of the participants
    		Args:
    			participants_path (str): Path of the participants list

    		Return:
    			A list with the names and twitter ids of the participants ->
    	"""
    participants_list = []
    with open(participants_path, 'r') as f:
        for participant in f:
            participant = participant.replace("\n", "").split(' @')
            participant[0] = file_utils.strip_accents(participant[0]) #strip accents...
            if (len(participant) > 1):
                participant[1] = "@" + participant[1]
            participants_list.append(participant)
    return participants_list


def load_bboxes_and_embs(root_input_path_bbox, root_input_path_embs, tertulianos_list=[]):
    bbox_total_faces = list()
    bbox_total_labels = list()
    embs_total_faces = list()
    embs_total_labels = list()
    if(tertulianos_list == []):
        tertulianos_list = [identity.replace(".npz","") for identity in os.listdir(root_input_path_embs)]
    for tertuliano in sorted(tertulianos_list):
        # BBOXES
        path_boundingbox = os.path.join(root_input_path_bbox,tertuliano + ".npz")
        # if (not(os.path.exists(path_boundingbox))):
        #     tertuliano = tertuliano.replace(" ", "_")
        #     path_boundingbox = os.path.join(root_input_path_bbox, eleccion_query, tertuliano + ".npz")
        if(os.path.exists(path_boundingbox)): #if path exists...
            print("Loading bbox&embs of: ", tertuliano, " ...")

            boundingbox_faces_total = list()
            boundingbox_labels_total = list()

            data = np.load(path_boundingbox, allow_pickle=True)
            faces, labels = data['arr_0'], data['arr_1']

            for face in faces:
                boundingbox_faces_total.append(face)

            for label in labels:
                boundingbox_labels_total.append(label)

            bbox_total_faces.extend(boundingbox_faces_total)
            bbox_total_labels.extend(boundingbox_labels_total)
            # EMBEDDINGS
            path_embedding = os.path.join(root_input_path_embs,tertuliano + ".npz")

            embedding_faces_totales = list()
            embedding_labels_totales = list()

            data = np.load(path_embedding, allow_pickle=True)
            faces, labels = data['arr_0'], data['arr_1']

            for face in faces:
                embedding_faces_totales.append(face)

            for label in labels:
                embedding_labels_totales.append(label)

            embs_total_faces.extend(embedding_faces_totales)
            embs_total_labels.extend(embedding_labels_totales)
        else:
            print(" TERTULIANO: ", tertuliano, " , bbox&embs NOT FOUND!")
    print('\nBOUNDINGBOXES')
    print('> Faces: \t' + str(len(bbox_total_faces)))
    print('> Labels: \t' + str(len(bbox_total_labels)))
    print('\nEMBEDDINGS')
    print('> Faces: \t' + str(len(embs_total_faces)))
    print('> Labels: \t' + str(len(embs_total_labels)))
    return bbox_total_faces, bbox_total_labels, embs_total_faces, embs_total_labels


def load_cluster(cluster_model_path):
    if (os.path.exists(cluster_model_path)):
        cluster = pickle.load(open(cluster_model_path, 'rb'))
        return cluster
    else:
        raise FileNotFoundError("Cluster not found in path: "+cluster_model_path)



def load_image(image_path, grayscale=False, target_size=None, colormode = "rgb"):
    pil_image = image.load_img(image_path, grayscale=grayscale, target_size=target_size, color_mode=colormode)
    return image.img_to_array(pil_image)


def load_best_parameters(input_dir,program, assignation_type, assignation_method):
    """
    Load best cluster paramenters (eps,min_samples for DBSCAN/ cluster_selection_epsilon,min_samples for HDBSCAN) for specific program
        :param program: Program name
        :return:
            - best min_sample (DBSCAN/HDBSCAN)
            - best eps (DBSCAN) or best cluster_selection_epsilon (HDBSCAN)
    """
    path_best_clustering_parameters = os.path.join(input_dir, "parameters", program,
                                                   assignation_type + "_"+assignation_method+"_best_possible_cluster_parameters.npz")
    cluster_param = np.load(path_best_clustering_parameters, allow_pickle=True)
    minSamples_eps = cluster_param["arr_0"]
    return int(minSamples_eps[0][0]), minSamples_eps[0][1]


def load_best_parameters_new(input_path_parameters, quality_metric="silhouette"):
    """
    Load best cluster paramenters (eps,min_samples for DBSCAN/ cluster_selection_epsilon,min_samples for HDBSCAN) for specific program
        :param program: Program name
        :param participant: Participant name od COMMON for generic cluster
        :return:
            - best min_sample (DBSCAN/HDBSCAN)
            - best eps (DBSCAN) or best cluster_selection_epsilon (HDBSCAN)
    """
    path_best_clustering_parameters = os.path.join(input_path_parameters,
                                                   quality_metric+"_best_cluster_parameters.npz")
    cluster_param = np.load(path_best_clustering_parameters, allow_pickle=True)
    first_second_params = cluster_param["arr_0"]
    return (first_second_params[0][0]), int(first_second_params[0][1])