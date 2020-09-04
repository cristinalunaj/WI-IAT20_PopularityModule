"""
    GUI for labelling information
	author: Cristina Luna
	date: 08/2020
	Usage:
		python3 interactive_error_GUI.py
        --input-path-imgs ../datasets/DATASET_GOOGLE_IMGS/download_name
        --input-path-clusters .../DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/supervised_results/BASELINE_OFICIAL_LN24H_SOLOIND_globalMax/Cfacial_clustering_results/Google/IND/numImgs/assignation_x/clustering_th_0/cluster_filtering_assignation/TOTAL/x_eps_0_min_samples_0/ZZdebug
        --output-path-answers .../datasets/DATASET_GOOGLE_IMGS/CUESTIONARIOS
        --name_form rtve2018
        --annotator_id CLJ
        --ref-imgs-path .../DATASET_RTVE_2018/RTVE2018DB/test/GENERATED_ME/DATASET_LN24H/frames
        --labels_ref_video .../DATASET_RTVE_2018/RTVE2018DB/test/rttm_FACE
	Options:
	    --input-path-imgs: Root path with the Google images downloaded
	    --input-path-clusters: Path to the clusters folder created from faces.
	    --output-path-answers: Path to save annotation results
	    --name-form: Name to include into generated annotation results.
	    --annotator-id: Name/id of the annotator.
	    --ref-imgs-path: Path where the reference images are or where the frames of the labelled videos are [OPTIONAL: Only for showing reference images]
		--labels_ref_video:If path of labelled video are provided, then this parameter is required for finding the labels [OPTIONAL: Only for showing reference images]

		-h, --help	Display script additional help
"""
from tkinter import *
from tkinter import ttk
import os, argparse
import pandas as pd
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import src.utils.files_utils as data_clean



class MyFirstGUI:
    def __init__(self, master,path_imgs,path_clt_users,users_id,annotator_id,output_dir,name_form, index_user = 0,ini_size=200, padding = 5, master_title = "A simple GUI", ref_imgs_dict={}):
        #PREVIOUS APP INITIALIZATION
        self.master = master
        master.title("Evaluating IDs")
        self.original_path_imgs = path_imgs
        self.original_path_clt = path_clt_users
        self.user_ids = users_id
        self.index_user = index_user
        self.group_size2show = 9
        self.annotator_id = annotator_id
        self.output_dir = output_dir
        self.name_form = name_form
        self.ref_imgs_dict = ref_imgs_dict
        #COMMON INITIALIZATION
        self.initialize_user_window()

        # INITIAL LAYOUT
        self.center_frame = Frame(self.master, width=800, height=400, bg='grey', borderwidth=0)
        self.center_frame.grid(row=0, column=0, padx=5, pady=5)
        self.left_frame = Frame(self.master, relief=tk.RAISED,
                                borderwidth=1, width=400, height=400)
        self.left_frame.grid(row=1, column=0, padx=5, pady=5)
        self.left_frame_down = Frame(self.master, relief=tk.FLAT,
                                borderwidth=1, width=400, height=400)
        self.left_frame_down.grid(row=2, column=0, padx=0, pady=0)
        self.right_frame_up = Frame(self.master, width=100, height=10, borderwidth=1)
        self.right_frame_up.grid(row=0, column=1, padx=5, pady=5)
        self.right_frame_middle = Frame(self.master, width=400, height=400, borderwidth=1)
        self.right_frame_middle.grid(row=1, column=1, padx=5, pady=5)


        # CENTER FRAME
        self.center_label = Label(self.center_frame,
                                  text="Select where IT IS NOT " + self.user_ids[self.index_user] +" -- Page ("+str(self.current_pack_index)+")")
        self.center_label.grid(row=0, column=0, padx=0, pady=0)

        # LEFT FRAME
        #UP - IMAGES
        self.left_mov_error_button = Button(self.left_frame, text="<", command=self.left_mov_error_button_callback)
        self.left_mov_error_button.grid(row=2, column=0, padx=padding + 3, pady=padding + 3)
        self.rigth_mov_error_button = Button(self.left_frame, text=">", command=self.rigth_mov_error_button_callback)
        self.rigth_mov_error_button.grid(row=2, column=4, padx=padding + 3, pady=padding + 3)
        self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=ini_size, padding=padding)
        # DOWN - NEXT & CLOSE BUTTONS
        self.next_id_button = Button(self.left_frame_down, text="NEXT ID", command=self.next_id_button_callback, bg = "grey")
        self.next_id_button.grid(row=5, column=0, padx=padding + 3, pady=padding + 3)
        self.close_button = Button(self.left_frame_down, text="SAVE & EXIT", command=self.close_button_callback, bg = "grey")
        self.close_button.grid(row=5, column=2, padx=padding + 3, pady=padding + 3)

        # RIGTH FRAME
        #UP: ID SELECTOR
        label_select_id = Label(master=self.right_frame_up, text = "Select ID to label:")
        label_select_id.grid(padx=padding + 3, pady=padding + 3, sticky=E)
        self.combobox_ids = ttk.Combobox(self.right_frame_up, values=users_id)
        self.combobox_ids.current(self.index_user)
        self.combobox_ids.grid(sticky=E, padx=padding + 3, pady=padding + 3)
        self.combobox_ids.bind("<<ComboboxSelected>>", self.combobox_users_callback)

        #MIDDLE - REF. IMAGE
        label_select_id_REF = Label(master=self.right_frame_middle, text="REFERENCE IMAGE")
        label_select_id_REF.grid(padx=padding + 3, pady=padding + 3, row=3, column=1)
        self.ref_img_extra_size = 60
        if(bool(ref_imgs_dict) and ref_imgs_dict[self.user_ids[self.index_user]]!=None):
            img_ref = Image.open(
                ref_imgs_dict[self.user_ids[self.index_user]])
            resized_img_ref = img_ref.resize((ini_size+self.ref_img_extra_size, ini_size+self.ref_img_extra_size))
            photo_ref = ImageTk.PhotoImage(master=self.master, image=resized_img_ref)
        else:
            black_img = Image.fromarray(np.zeros((ini_size+self.ref_img_extra_size, ini_size+self.ref_img_extra_size, 3), dtype="uint8"), 'RGB')
            photo_ref = ImageTk.PhotoImage(master=self.master, image=black_img)
        self.lbl_img = Label(master=self.right_frame_middle, image=photo_ref)
        self.lbl_img.image = photo_ref
        self.lbl_img.grid(row=2, column=1, padx=padding + 3, pady=padding + 3)  #,
        # self.left_mov_ref_button = Button(self.right_frame_middle, text="<", command=self.greet)
        # self.left_mov_ref_button.grid(row=2, column=0, padx=padding + 3, pady=padding + 3)
        # self.rigth_mov_ref_button = Button(self.right_frame_middle, text=">", command=self.greet)
        # self.rigth_mov_ref_button.grid(row=2, column=2, padx=padding + 3, pady=padding + 3)


    def combobox_users_callback(self, event):
        print(event.widget.get()," selected")
        self.index_user = event.widget.current()
        self.initialize_user_window()
        self.update_view(padding=5, ini_size=200)


    def update_view(self, padding = 5, ini_size = 200):
        self.center_label.config(text="Select where IT IS NOT " + self.user_ids[self.index_user]+" -- Page ("+str(self.current_pack_index)+")")
        self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=ini_size, padding=padding)
        #REFERENCE IMG
        self.fill_in_ref_img(ini_size)


    def fill_in_ref_img(self, ini_size = 200, padding = 5):
        if (bool(self.ref_imgs_dict) and self.ref_imgs_dict[self.user_ids[self.index_user]] != None):
            img_ref = Image.open(
                self.ref_imgs_dict[self.user_ids[self.index_user]])
            resized_img_ref = img_ref.resize((ini_size+self.ref_img_extra_size, ini_size+self.ref_img_extra_size))
            photo_ref = ImageTk.PhotoImage(master=self.master, image=resized_img_ref)
        else:
            black_img = Image.fromarray(np.zeros((ini_size+self.ref_img_extra_size, ini_size+self.ref_img_extra_size, 3), dtype="uint8"), 'RGB')
            photo_ref = ImageTk.PhotoImage(master=self.master, image=black_img)
        self.lbl_img = Label(master=self.right_frame_middle, image=photo_ref)
        self.lbl_img.image = photo_ref
        self.lbl_img.grid(row=2, column=1, padx=padding + 3, pady=padding + 3)


    def save_id(self):
        print("Saving ", self.user_ids[self.index_user], " data ...")
        cuestionario_df = pd.DataFrame([[self.annotator_id, self.user_ids[self.index_user], self.TP, self.FP, self.TN, self.FN]],
                                                              columns=["ID_anotador", "Nombre_personaje", "TP", "FP",
                                                                       "TN", "FN"])
        output_path = os.path.join(self.output_dir,
                                  self.name_form + "_" + self.annotator_id + "_" + "{0:0=5d}".format(self.index_user) + ".csv")
        cuestionario_df.to_csv(output_path, index=False, sep=";", header=True)

    def close_button_callback(self):
        self.save_id()
        self.master.quit()

    def initialize_user_window(self):
        self.current_pack_index = 0
        self.in_user_imgs = True
        path_clts = os.path.join(self.original_path_clt, self.user_ids[self.index_user].replace("_", " "))
        self.packs_of_imgs_user, self.packs_of_imgs_noise = self.create_imgs_packs(self.original_path_imgs, path_clts,
                                                                   self.user_ids[self.index_user],group_size=self.group_size2show)
        self.max_packs_user = len(self.packs_of_imgs_user)
        self.max_packs_noise = len(self.packs_of_imgs_noise)
        self.TP = list(set([item for t in self.packs_of_imgs_user for item in t]).difference(set([None])))
        self.FP = []
        self.TN = list(set([item for t in self.packs_of_imgs_noise for item in t]).difference(set([None])))
        self.FN = []

    def next_id_button_callback(self):
        self.save_id()
        if(self.index_user+1<len(self.user_ids)):
            self.index_user += 1
            self.initialize_user_window()
            self.update_view(padding=5, ini_size=200)

    def left_mov_error_button_callback(self):
        if (self.in_user_imgs): #USER
            if (self.current_pack_index-1<0): return
            self.current_pack_index-=1
            self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=200, padding=5)
            self.center_label.config(
                text="Select where IT IS NOT " + self.user_ids[self.index_user] + " -- Page (" + str(
                    self.current_pack_index) + ")")
        else: #NOISE
            if (self.current_pack_index-1<0):
                self.in_user_imgs = True
                self.current_pack_index = self.max_packs_user-1
                self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(text="Select where IT IS NOT " + self.user_ids[self.index_user] +" -- Page ("+str(self.current_pack_index)+")")
            else:
                self.current_pack_index-=1
                self.fill_in_left_frame(self.packs_of_imgs_noise[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(
                    text="Select where IT IS " + self.user_ids[self.index_user] + " -- Page (" + str(
                        self.current_pack_index) + ")")

    def rigth_mov_error_button_callback(self):
        if(self.in_user_imgs):
            if (self.current_pack_index+1 >= self.max_packs_user and len(self.packs_of_imgs_noise)>0):
                self.in_user_imgs = False
                self.current_pack_index = 0
                self.fill_in_left_frame(self.packs_of_imgs_noise[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(text= "Select where IT IS " + self.user_ids[self.index_user] +" -- Page ("+str(self.current_pack_index)+")")
            elif(self.current_pack_index+1 >= self.max_packs_user and len(self.packs_of_imgs_noise)<=0):return
            else:
                self.current_pack_index+=1
                self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(
                    text="Select where IT IS NOT " + self.user_ids[self.index_user] + " -- Page (" + str(self.current_pack_index) + ")")
        else:
            if (self.current_pack_index + 1 >= self.max_packs_noise): return
            self.current_pack_index += 1
            self.fill_in_left_frame(self.packs_of_imgs_noise[self.current_pack_index], ini_size=200, padding=5)
            self.center_label.config(
                text="Select where IT IS " + self.user_ids[self.index_user] + " -- Page (" + str(
                    self.current_pack_index) + ")")


    def fill_in_left_frame(self, img_paths, ini_size=200, padding = 5):
        n = 0
        self.map_button_img = []
        for i in range(1,4):
            #self.master.columnconfigure(i, weight=1, minsize=75)
            #self.master.rowconfigure(i, weight=1, minsize=50)
            for j in range(1, 4):
                scnd_left_frame = tk.Frame(
                    master=self.left_frame,
                    relief=tk.FLAT,
                    borderwidth=1
                )
                scnd_left_frame.grid(row=i, column=j, padx=padding, pady=padding)
                if(img_paths[n]!=None):
                    img_loaded = Image.open(img_paths[n])
                    resized_img = img_loaded.resize((ini_size,ini_size))
                    photo1 = ImageTk.PhotoImage(master=self.master,image=resized_img)
                    if(img_paths[n] in self.TP or img_paths[n] in self.TN):
                        bg_color = "green"
                    else:
                        bg_color = "red"
                    btn = Button(master=scnd_left_frame, image=photo1, bg=bg_color, width=ini_size, height=ini_size,
                                 relief="flat", padx=0,pady=0,overrelief="flat", highlightthickness=0, borderwidth=padding,
                                 text = img_paths[n]) #cursor="question_arrow", t
                    btn.image = photo1
                    btn.bind('<Button-1>', self.change_bg)
                    btn.bind('<Button-3>', self.zoom)
                else:
                    black_img = Image.fromarray(np.zeros((ini_size, ini_size, 3), dtype="uint8"), 'RGB')
                    default_black_img = ImageTk.PhotoImage(master=self.master, image=black_img)
                    btn = Button(master=scnd_left_frame, image=default_black_img)
                    btn.image = default_black_img
                btn.pack(padx=padding, pady=padding,  side = RIGHT)
                n+=1

    def change_bg(self,event):
        #TO DO: SELECT CORRECT & INCORRECT IMGS
        if(event.widget["bg"]=="red"):
            event.widget.config(bg="green")
            if (self.in_user_imgs):
                self.FP = list(set(self.FP).difference(set([event.widget["text"]])))
                self.TP.append(event.widget["text"])
            else: #NOISE
                self.FN = list(set(self.FN).difference(set([event.widget["text"]])))
                self.TN.append(event.widget["text"])
        elif(event.widget["bg"]=="green"):
            event.widget.config(bg="red")
            if (self.in_user_imgs):
                self.TP = list(set(self.TP).difference(set([event.widget["text"]])))
                self.FP.append(event.widget["text"])
            else: #NOISE
                self.TN = list(set(self.TN).difference(set([event.widget["text"]])))
                self.FN.append(event.widget["text"])


    def zoom(self, event):
        newWindow = tk.Toplevel(root)
        img_loaded = Image.open(event.widget["text"])
        photo1 = ImageTk.PhotoImage(master=newWindow, image=img_loaded)
        lab = Label(newWindow,
              text="Zoom img",image = photo1)
        lab.image = photo1
        lab.pack(padx=1, pady=1, side=TOP)



    def create_imgs_packs(self, path_imgs, path_clts, user_id, group_size= 8):
        supposed_imgs_with_user, supposed_imgs_without_user = self.select_img_paths_of_user_vs_noise(path_clts, user_id)
        # Create complete list of imgs with path:
        list_imgs_user_clt = sorted([os.path.join(path_imgs, name_img) for name_img in supposed_imgs_with_user])
        list_noise_user_clt = sorted([os.path.join(path_imgs, name_img) for name_img in supposed_imgs_without_user])
        packs_of_imgs_user = (list(zip(*(iter(list_imgs_user_clt),) * group_size)))
        #TO CHECK!!!
        if(len(list_imgs_user_clt)%group_size!=0):
            last_tuple = list_imgs_user_clt[len(packs_of_imgs_user)*group_size::]
            last_tuple += [None] * (group_size - len(last_tuple))
            #non_in_tuple = list_imgs_user_clt[-(len(list_imgs_user_clt)%group_size)::]+[None*(group_size-len(list_imgs_user_clt[-(len(list_imgs_user_clt)%group_size)::]))]
            packs_of_imgs_user.append(tuple(last_tuple))
        #Add extra imgs
        packs_of_imgs_noise = (list(zip(*(iter(list_noise_user_clt),) * group_size)))
        if (len(list_noise_user_clt) % group_size != 0):
            last_tuple = list_noise_user_clt[len(packs_of_imgs_noise) * group_size::]
            last_tuple += [None] * (group_size - len(last_tuple))
            packs_of_imgs_noise.append(tuple(last_tuple))
        return packs_of_imgs_user, packs_of_imgs_noise



    def select_img_paths_of_user_vs_noise(self,input_path_clt, user_id):
        input_path_clt_embs = os.path.join(input_path_clt, "embeddings_sum")
        list_clusters = os.listdir(input_path_clt_embs)
        noise_clt = list_clusters
        for clt_name in list_clusters:
            if (user_id.replace(" ", "_") in clt_name):
                user_assigned_clt = clt_name
                noise_clt.remove(clt_name)
        user_key_imgs_data = np.load(os.path.join(input_path_clt_embs, user_assigned_clt), allow_pickle=True)
        imgs_with_user = set(user_key_imgs_data["arr_1"])
        imgs_without_user = set()
        for noise in noise_clt:
            embs_noise = np.load(os.path.join(input_path_clt_embs, noise), allow_pickle=True)
            imgs_without_user = imgs_without_user.union(set(embs_noise["arr_1"]))
        imgs_without_user = imgs_without_user.difference(imgs_with_user)
        return imgs_with_user, imgs_without_user



def get_ref_imgs(list_users, frames_path, labels_path="", fps = 1):
    """
    Module for loading reference images. Modify to adapt to the specific task.
    :param list_users: List with the names of participants to annotate
    :param frames_path: Path to the frames of the videos.
    :param labels_path: Path to the labels of the videos that indicate where the participant appears.
    :param fps: frames per second at wich the frames where extracted
    :return: Dict. with participants as key and reference img. as value
    """
    dict_ref_imgs = {}
    if(not labels_path == ""):
        for video_id in os.listdir(labels_path):
            df_labels = pd.read_csv(os.path.join(labels_path, video_id), sep=" ", header=None)
            participants = list(set(df_labels[7].values))
            video_id = video_id.rsplit("-",1)[0]
            for part in participants:
                df_part = df_labels.loc[df_labels[7] == part]
                for i in range(1, len(df_part)):
                    t_ini = float(df_labels.loc[df_labels[7]==part].iloc[i][3])
                    dur = float(df_labels.loc[df_labels[7]==part].iloc[i][4])
                    if(dur > 1):
                        num_frame = int(t_ini*fps)+2
                        frame_name = os.path.join(frames_path, video_id, "OTHER", video_id+"_"+"{0:0=6d}".format(num_frame)+".png")
                        if(os.path.exists(frames_path)):
                            part = data_clean.strip_accents(part)
                            dict_ref_imgs[part] = frame_name
                            break
                    else:
                        print("debug")
        #Ckeck that all participants have their ref img:
        for id in list_users:
            if(not id in dict_ref_imgs):
                dict_ref_imgs[id] = None
    else:
        df_ref_imgs = pd.DataFrame([os.listdir(frames_path)], columns=["ref_img"])
        for part in list_users:
            dict_ref_imgs[part] = df_ref_imgs[df_ref_imgs['ref_img'].str.contains(part)]
    return dict_ref_imgs




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-path-imgs", required=True,
        help="path to input directory of images")
    ap.add_argument("--input-path-clusters", required=True,
                    help="path to clusters directory of images")
    ap.add_argument("-o", "--output-path-answers", required=True,
        help="path to input directory of images")
    ap.add_argument("--name-form",
        help="path to input directory of images", default="")
    ap.add_argument("--annotator-id",
                    help="path to input directory of images", default="")
    ap.add_argument("--reference-imgs-path",
                    help="path to input directory of images", default="")
    ap.add_argument("--ref-imgs-path",
                    help="Path where the reference images are or where the frames of the labelled video are (OPTIONAL: Only for reference images)", default="")
    ap.add_argument("--labels-ref-video",
                    help="If path of labelled video are provided, then this parameter is required for finding the labels (OPTIONAL: Only for reference images)",
                    default="")

    args = vars(ap.parse_args())
    input_path_imgs = args["input_path_imgs"]
    input_path_clusters = args["input_path_clusters"]
    output_path_questions = args["output_path_answers"]
    name_form = args["name_form"]
    ID_anotador = args["annotator_id"]
    frames_path = args["ref_imgs_path"]
    labels_path = args["labels_ref_video"]

    # # -------------------------------------------
    os.makedirs(output_path_questions, exist_ok=True)
    list_users = sorted(os.listdir(input_path_imgs))
    # check if previous attempts:
    list_output_files = sorted(os.listdir(output_path_questions))
    ini_name = name_form + "_" + ID_anotador
    possible_files = []
    for filename in sorted(list_output_files):
        if (ini_name in filename):
            possible_files.append(filename)
    if (len(possible_files) > 0):
        cuestionario_df = pd.read_csv(os.path.join(output_path_questions, possible_files[-1]), sep=";", header=0)
        index_n_users = int(possible_files[-1].split(".csv")[0].split("_")[-1])
        print("Last annotated ID in previous session: ", list_users[index_n_users])
        index_n_users += 1
    else:
        # if new user:
        index_n_users = 0

    #-------- REFERENCE IMGS args ------- MODULE FOR showing the reference images on the GUI
    if(frames_path or labels_path):
        dict_ref_imgs = get_ref_imgs(list_users, frames_path, labels_path=labels_path)
    else:
        dict_ref_imgs = {}



    #OPEN NEW APP WINDOW
    root = Tk()
    my_gui = MyFirstGUI(root,input_path_imgs,input_path_clusters,list_users, ID_anotador, output_path_questions, name_form, index_n_users, ref_imgs_dict=dict_ref_imgs)
    root.mainloop()
    #root.destroy()