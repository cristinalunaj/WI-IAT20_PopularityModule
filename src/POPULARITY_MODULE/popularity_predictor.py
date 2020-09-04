import pandas as pd
import subprocess, os
import src.utils.loader as loader

def create_test_arff(participant, test_df, aux_path):
    arff_text = "@relation summary_features \n\n" \
                "@attribute n_faces numeric\n" \
                "@attribute avg_confidence_faces numeric\n" \
                "@attribute std_confidence_faces numeric\n" \
                "@attribute avg_relativeSize_faces numeric\n" \
                "@attribute std_relativeSize_faces numeric\n" \
                "@attribute avg_thirdRule_x numeric\n" \
                "@attribute std_thirdRule_x numeric\n" \
                "@attribute avg_thirdRule_y numeric\n" \
                "@attribute std_thirdRule_y numeric\n" \
                "@attribute num_clts numeric\n" \
                "@attribute avg_silhouette numeric\n" \
                "@attribute avg_intra_clt_dist numeric\n" \
                "@attribute avg_inter_clt_dist numeric\n" \
                "@attribute faces_in_noise_clt numeric\n" \
                "@attribute num_core_samples numeric\n" \
                "@attribute avg_imgs_clt numeric\n" \
                "@attribute avg_std_silhouette numeric\n" \
                "@attribute avg_std_intra_clt_dist numeric\n" \
                "@attribute avg_std_inter_clt_dist numeric\n" \
                "@attribute avg_n_core_samples numeric\n" \
                "@attribute std_n_core_samples numeric\n" \
                "@attribute GTrends_popularity numeric\n" \
                "@attribute label {1,0}\n\n" \
                "@data\n"

    data = test_df.loc[test_df["id"]==participant]
    data = data.drop(columns="id")
    data_str = ""
    for ele in data.values[0]:
        data_str += str(ele)+","
    data_str = data_str[0:-3]
    arff_text+=data_str
    print(arff_text)

    f = open(aux_path, "w")
    f.write(arff_text)

def evaluate_test_arff(model_path, test_arff_path, out_path):
    """
    Obtain predictions of test_file using the trained model in model_path
    :param output_folder:
    :param output_name:
    :param model_path:
    :param test_file:
    """
    # PREDICTIONS FILE HEADERS: INSTANCE, ACTUAL, PREDICTED, ERROR
    bash_file_path = "../../data/bash_scripts/explorer_test_model.sh  "
    with open(out_path, 'w') as fi:
        fi.close()
    command = "".join([bash_file_path, test_arff_path, " ", model_path, " ", out_path])
    print(command)
    subprocess.call(command, shell=True)
    remove_lines(out_path)  # remove headers of prediction file
    df_participant = pd.read_csv(out_path, header=0, sep=",")
    return df_participant

def remove_lines(path_csv):
    with open(path_csv, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(path_csv, 'w') as fout:
        fout.writelines(data[4:]) #en 4 las cabeceras
        fout.close()


if __name__ == "__main__":
    th = "05"
    path_model = "../../data/models/popularity_module/CLASIF/th"+th+"/RandomForest.model"
    complete_df_ids = "../../data/datasets/popularity_module_features/train/summary_features_participants_classification_th"+th+".csv"
    aux_path = "../../data/datasets/popularity_module_features/aux_test.arff"
    out_path_prediction = "../../data/datasets/popularity_module_features/aux_prediction.csv"
    complete_df = pd.read_csv(complete_df_ids, header=0, sep=",")
    bash_test_model = ""
    path_participants = "../../data/datasets/DATASET_GOOGLE_IMGS/participants/"
    list_participants = loader.load_list_of_tertulianos(path_participants, "participants_complete_rtve2018",".csv")
    #list_participants = [participant.replace(" ", "_") for participant in part]
    df_popularity = pd.DataFrame([], columns=["prediction", "popular", "id"])
    out_path_popularity_df = "../../data/results/popularity_models_output/popularity_df_th"+th+".csv"
    for participant in list_participants:
        participant = participant.replace("_", " ")
        create_test_arff(participant, complete_df, aux_path)
        df_participant = evaluate_test_arff(path_model, aux_path, out_path_prediction)
        df_popularity = df_popularity.append(pd.DataFrame([[df_participant["predicted"][0].split(":")[-1], df_participant["predicted"][0].split(":")[-1]=="1", participant
        ]], columns=["prediction", "popular", "id"]))
    df_popularity.to_csv(out_path_popularity_df, sep=";", header=True, index=False)





