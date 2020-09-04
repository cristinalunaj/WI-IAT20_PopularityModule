"""
    Change structure of folders in order to have a common pattern for next stages:
    output_dir/
        |_program/
            |_embs/
            |_bbox/
            |_...
                |_ID0/
                    |_ *.npz
                |_ID1
                    |_*.npz
                |_ ...
            |_embs_sum/
                |_ID0.npz
                |_ID1.npz
                |_...
    author: Cristina Luna, Ricardo Kleinlein
    date: 05/2020
    Usage e.g:
        python3 RefactorDS.py
        --root-input-path ~/DATASET_GOOGLE_IMGS/download_name
        --set-to-analyse Google
        --program-participants-folder /~/DATASET_LN24H/tertulianos_ids
        --output-dir ~/DATASET_GOOGLE_IMGS/refactor_DS

    Options:
        --root-input-path Path to the root folder whose sub-folders need to be refactored/renamed
        --set-to-analyse Refactor frames structure in order to be in concordance with OCR structure: "
                 program0/
                   |_participant0/
                       |_participant0_frame0.png
                      |_participant0_frame1.png
                       |...
                   |_participant1/"
                   ...
                 In PROGRAM there is one unique participant with label OTHER
                 **Options: (Google, PROGRAM)
        --program-participants-folder Path to the folder with the .csv that contain the names of the
                 participants per program (See /AQuery_launcher/Aqeury_launcher_bingEngine.py
        --output-dir Directory to save results in

"""

import os, time
import src.utils.loader as loader
from distutils.dir_util import copy_tree
from src.BaseArgs import RefactorDSArgs


class RefactorDS:

    def __init__(self, set_to_analyse, root_input_path, output_dir, path_participants_folder):
        self.set_to_analyse = set_to_analyse  # Google, OCR, PROGRAM or PROGRAM_OTHER
        self.root_input_path = root_input_path
        self.output_dir = output_dir
        self.program_names = [file.split(".")[0] for file in os.listdir(path_participants_folder)]
        self.program_file_extension = os.listdir(root_input_path)[0].split(".")[-1] \
            if ("." in os.listdir(root_input_path)[0]) else ""
        self.path_participants_folder = path_participants_folder

    def refactor_google(self):
        """
        Refactor Google DS forlder dividing IDs per program
        """
        for program in self.program_names:
            input_participants_path = os.path.join(self.path_participants_folder, program + self.program_file_extension)
            participant_names = [val.replace(" ","_") for sublist in loader.get_participants(input_participants_path) for val in sublist]
            for participant in participant_names:
                # check if participant in downloaded participants from Google:
                input_participant_google_ds = os.path.join(self.root_input_path, participant)
                output_participants_in_program_google = os.path.join(self.output_dir, program, participant)
                # copy images from input_folder to output_folder:
                if os.path.isdir(input_participant_google_ds):
                    copy_tree(input_participant_google_ds, output_participants_in_program_google)

    def refactor_program(self, new_folder_name="OTHER"):
        """
        Refactor Program folder addding an ID "new_folder_name" as the ID that represents all the participants
        in the program
        :param new_folder_name: subfolder ID for program
        """
        for program in self.program_names:
            program_input_path = os.path.join(self.root_input_path, program)
            program_output_path = os.path.join(self.root_input_path, program, new_folder_name)
            os.makedirs(program_output_path, exist_ok=True)
            intial_n_imgs = len(os.listdir(program_input_path))-1
            os.system("mv " + program_input_path + "/*.png " + program_output_path)
            while (len(os.listdir(program_output_path)) < intial_n_imgs):
                time.sleep(15) #15 seg

    def run_refactor(self):
        """
        Run refactor
        """
        if self.set_to_analyse == "Google":
            self.refactor_google()
        elif self.set_to_analyse == "PROGRAM":
            self.refactor_program()


if __name__ == "__main__":
    refactor_args_obj = RefactorDSArgs()
    args = refactor_args_obj.parse()
    refactorDS_obj = RefactorDS(args.set_to_analyse, args.root_input_path,
                                args.output_dir, args.program_participants_folder)
    refactorDS_obj.run_refactor()
