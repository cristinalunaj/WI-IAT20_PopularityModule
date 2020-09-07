"""
	Extract participants list per program, also reformat label files to md_eval library format.
	(https://github.com/usnistgov/SCTK  - version. 2.4.10).
	A dict object is created in order to have a dictionary with the mappings between participan names in labels
	and participant names after removing accents and problematic characters.
	author: Cristina Luna,Ricardo Kleinlein
	date: 06/2020
	Usage:
		(e.g. python3 DSPreparator.py
        --input-path-labels-rttm ~/RTVE2018DB/test/rttm
        --output-path-labels-rttm ~/RTVE2018DB/test/rttm_INFO_v2/Amd_eval_adapted_rttm
        --output-path-participants-folder ~/RTVE2018DB/test/rttm_INFO_v2
        --category FACEREF
        --programs-folder ~/RTVE2018DB/test/video
	Options:
		--input-path-labels-rttm: Path to given labels as .rttm. Expected labels file names: program-category.rttm
		 (e.g. expected input : LM-20170103-FACEREF.rttm)
        --output-path-labels-rttm: Path to save adapted labels to md_eval library (for extracting DER)
        (e.g. expected output : LM-20170103-FACEREF.rttm)
        --output-path-participants-folder: Path to generated files with participant names
        (e.g. expected output file (NO EXTENSION): LM-20170103  )
        --category: Reference category to process (FACEREF, SPKREF...)
        --programs-folder: Path to program videos (from their names we will extract the name of the programs)
        (e.g. expected video name (*Any extension): LM-20170103.* )
		--quiet	Hide visual information
		-h, --help	Display script additional help
"""

import os
import pandas as pd
import numpy as np
import src.utils.files_utils as data_clean
from src.BaseArgs import DSPreparatorArgs


class DSPreparator():

    def __init__(self, input_path_labels_rttm, category, output_dir_rttm_labels, output_dir_participants_files):
        self.input_path_labels_rttm = input_path_labels_rttm  # Google, OCR, PROGRAM or PROGRAM_OTHER
        self.category = category
        self.output_dir_rttm_labels = output_dir_rttm_labels
        self.output_dir_participants_files = output_dir_participants_files


    def get_participants_summary_nonSupervised(self, program, extension = ""):
        """
        Get files with the participants of the program from labels path.
        Names generated will be filtered and cleaned in order to remove accents and other marks that could cause problems
        during processing of creation of folders
        (e.g of conversion: José_Hervás -> Jose Hervas)
        :param program: Name of program
        :return:
            -set of participant with stripped accents
            -set of participants without filtering accents (as in original file)
            -dictionary with keys (participants with stripped accents) and values (original participant names)
        """
        input_participants_path = os.path.join(self.input_path_labels_rttm, program + "-" + self.category+extension)
        df_participants_category = pd.read_csv(input_participants_path, header=None)
        set_of_detected_participants = set()
        set_of_detected_participants_with_accents = set()
        dict_filtered_non_filtered_ids = {}
        for index, row in df_participants_category.iterrows():
            label_not_filtered = row[0]
            label = data_clean.strip_accents(row[0].replace("_", " "))
            # Create dict with the original name in order to recover these labels at the end during evaluation
            set_of_detected_participants = set_of_detected_participants.union(set([label]))
            set_of_detected_participants_with_accents = set_of_detected_participants_with_accents.union(
                set([label_not_filtered]))
            dict_filtered_non_filtered_ids[data_clean.strip_accents(row[0])] = label_not_filtered
        # Create file of participants in program
        self.save_participants_data(program, set_of_detected_participants)
        return set_of_detected_participants, set_of_detected_participants_with_accents, dict_filtered_non_filtered_ids

    def save_participants_data(self, program,set_of_detected_participants):
        """
        Save participants data into folders
        :param program:
        :param set_of_detected_participants:
        """
        os.makedirs(os.path.join(self.output_dir_participants_files, self.category, "participants"), exist_ok=True)
        output_participants_path = os.path.join(self.output_dir_participants_files, self.category, "participants",
                                                program)  # + ".csv")
        pd.DataFrame(sorted(list(set_of_detected_participants)), columns=[self.category]).to_csv(
            output_participants_path, header=False, index=False)


    def get_participants_summary_from_rttm(self, program):
        """
        Get files with the participants of the program from labels rttm.
        Names generated will be filtered and cleaned in order to remove accents and other marks that could cause problems
        during processing of creation of folders
        (e.g of conversion: José_Hervás -> Jose Hervas)
        :param program: Name of program
        :return:
            -set of participant with stripped accents
            -set of participants without filtering accents (as in original file)
            -dictionary with keys (participants with stripped accents) and values (original participant names)
        """
        input_rttm_path = os.path.join(self.input_path_labels_rttm, program + "-" + self.category + ".rttm")
        df_rttm_category = pd.read_csv(input_rttm_path,header=None,sep=" ")
        set_of_detected_participants = set()
        set_of_detected_participants_with_accents = set()
        dict_filtered_non_filtered_ids = {}
        for index, row in df_rttm_category.iterrows():
            info = row[0]
            #remove accents and weird punctuation from participant name
            label = data_clean.strip_accents(row[7].replace("_", " "))
            label_not_filtered = row[7]
            # .rttm format expected has information about participants at the beginning
            if("INFO" in info):
                #Create dict with the original name in order to recover these labels at the end during evaluation
                set_of_detected_participants = set_of_detected_participants.union(set([label]))
                set_of_detected_participants_with_accents = set_of_detected_participants_with_accents.union(set([label_not_filtered]))
                dict_filtered_non_filtered_ids[data_clean.strip_accents(row[7])] = label_not_filtered
            else:
                break
        # Create file of participants in program
        self.save_participants_data(program, set_of_detected_participants)
        return set_of_detected_participants,set_of_detected_participants_with_accents,dict_filtered_non_filtered_ids




    def adapt_rttm_labels_2DER(self, program, info_text = "SPKR-INFO", info_data = "SPEAKER"):
        """
        Change DER format in labels in order to change 'category' by SPEAKER and 'category-INFO' by SPKR-INFO
        since these are the keywords that md_eval can recognise for extracting DER
        :param program: Name program
        :param info_text: First column info expected to introduce users in the .rttm
        (e.g. of rttm output info row: SPKR-INFO LM-20170103 1 <NA> <NA> <NA> unknown Veronica_Guerrero <NA> )
        (e.g. of rttm input info row:  FACE-INFO LM-20170103 1 <NA> <NA> <NA> unknown Veronica_Guerrero <NA> )
        :param info_data: First column name expected to refer predictions
        (e.g. of rttm output data row: SPEAKER LM-20170103 1 930.676000 61.293000 <NA> <NA> Veronica_Guerrero <NA>)
        (e.g. of rttm input data row:     FACE LM-20170103 1 930.676000 61.293000 <NA> <NA> Veronica_Guerrero <NA>)
        :return: dataframe modified to md_eval correct output
        """
        input_rttm_path = os.path.join(self.input_path_labels_rttm, program + "-" + self.category + ".rttm")
        output_path_rttm_new = os.path.join(self.output_dir_rttm_labels, self.category, program + "-" + self.category + ".rttm")
        #Load labels rttm
        df_rttm_category_input = pd.read_csv(input_rttm_path, header=None, sep=" ")
        df_rttm_category_output = pd.DataFrame([], columns=["category", "program", "one", "t_ini", "duration", "nuloA", "nuloB","id", "nuloC"])
        for index, row in df_rttm_category_input.iterrows():
            if("INFO" in row[0]):
                df_rttm_category_output = df_rttm_category_output.append(pd.DataFrame([[info_text]+list(row[1::].values)], columns=["category", "program", "one", "t_ini", "duration", "nuloA", "nuloB","id", "nuloC"]))
            else:
                df_rttm_category_output = df_rttm_category_output.append(pd.DataFrame([[info_data] + list(row[1::].values)], columns=["category", "program", "one", "t_ini", "duration","nuloA", "nuloB", "id", "nuloC"]))
        #save df:
        os.makedirs(os.path.join(self.output_dir_rttm_labels, self.category),exist_ok=True)
        df_rttm_category_output.to_csv(output_path_rttm_new, sep=' ', index=False, header = False)
        return df_rttm_category_output


    def main_prepare_participant_files(self, list_programs, semi_supervised=True):
        """
        Extract modified version of the rttm labels (to fit md_eval format) and list of participants for each program
        :param list_programs: List with the names of the programs to analyse
        """
        dict_pred_labels_names = {}
        participants_set_filtered = set()
        for program in list_programs:
            if(semi_supervised):
                info_set, _, dict_non_filtrered = self.get_participants_summary_from_rttm(program)
            else:
                info_set, _, dict_non_filtrered = self.get_participants_summary_nonSupervised(program)
            participants_set_filtered = participants_set_filtered.union(info_set)
            #Create dict with keys:participants as predictions ,values:participants as in rttm
            for k in dict_non_filtrered:
                dict_pred_labels_names[k] = dict_non_filtrered[k]
            if(semi_supervised):
                self.adapt_rttm_labels_2DER(program)
        # SAVE COMPLETE SET FOR ALL PROGRAMS:
        output_rttm_path_complete = os.path.join(self.output_dir_participants_files, self.category, "participants_complete.csv")
        pd.DataFrame(sorted(list(participants_set_filtered)), columns=[self.category]).to_csv(output_rttm_path_complete, header=False, index=False)
        self.save_participants_data("TOTAL", sorted(list(participants_set_filtered)))


        # save dict
        output_path_dict = os.path.join(self.output_dir_participants_files, self.category, "complete_dict.npy")
        np.save(output_path_dict, dict_pred_labels_names)



if __name__ == "__main__":
    dsPreparator_args = DSPreparatorArgs()
    args = dsPreparator_args.parse()
    dsPrep_obj = DSPreparator(args.input_path_labels_rttm, args.category,
                              args.output_path_labels_rttm, args.output_path_participants_folder)
    info_set_complete_not_filtered = set()
    list_programs = os.listdir(args.programs_folder) if(args.programs_folder) else [args.programs_folder]
    list_programs_non_extension = [file_name.split(".")[0] for file_name in list_programs]
    dsPrep_obj.main_prepare_participant_files(list_programs_non_extension, args.semi_supervised)

















