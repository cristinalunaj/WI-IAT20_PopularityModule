"""
    GUI for validate labelled information [Only in a visual way, for re-annotations use interactive_error_GUI.py]
	author: Cristina Luna
	date: 08/2020
	Usage:
		python3 interactive_error_GUI_validation.py
        --input-path-imgs ../DATASET_GOOGLE_IMGS/download_name
        --output-path-answers .../DATASET_GOOGLE_IMGS/CUESTIONARIOS
	Options:
	    --input-path-imgs: Root path with the Google images downloaded
	    --output-path-answers: Path of the annotated results to validate
		-h, --help	Display script additional help
"""


from tkinter import *
from tkinter import ttk
import os, argparse
import pandas as pd
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
import ast



class validationGUI:
    def __init__(self, master,path_imgs,path_csv,index_user = 0,ini_size=200, padding = 5, master_title = "A simple GUI"):
        #PREVIOUS APP INITIALIZATION
        self.master = master
        master.title("Evaluating IDs")
        self.complete_df = pd.read_csv(path_csv, sep=";", header=0)

        self.original_path_imgs = path_imgs
        self.user_ids = sorted(self.complete_df["Nombre_personaje"])
        self.index_user = index_user
        self.group_size2show = 9

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
                                  text="Imágenes en las que ESTÁ " + self.user_ids[self.index_user] +" -- Page ("+str(self.current_pack_index)+")")
        self.center_label.grid(row=0, column=0, padx=0, pady=0)

        # LEFT FRAME
        #UP - IMAGES
        self.left_mov_error_button = Button(self.left_frame, text="<", command=self.left_mov_error_button_callback)
        self.left_mov_error_button.grid(row=2, column=0, padx=padding + 3, pady=padding + 3)
        self.rigth_mov_error_button = Button(self.left_frame, text=">", command=self.rigth_mov_error_button_callback)
        self.rigth_mov_error_button.grid(row=2, column=4, padx=padding + 3, pady=padding + 3)
        self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=ini_size, padding=padding)
        # DOWN - NEXT & CLOSE BUTTONS
        self.next_id_button = Button(self.left_frame_down, text="SIGUIENTE ID", command=self.next_id_button_callback, bg = "grey")
        self.next_id_button.grid(row=5, column=0, padx=padding + 3, pady=padding + 3)
        self.close_button = Button(self.left_frame_down, text="GUARDAR & SALIR", command=self.close_button_callback, bg = "grey")
        self.close_button.grid(row=5, column=2, padx=padding + 3, pady=padding + 3)

        # RIGTH FRAME
        #UP: ID SELECTOR
        label_select_id = Label(master=self.right_frame_up, text = "Selecciona la ID a etiquetar:")
        label_select_id.grid(padx=padding + 3, pady=padding + 3, sticky=E)
        self.combobox_ids = ttk.Combobox(self.right_frame_up, values=self.user_ids)
        self.combobox_ids.current(self.index_user)
        self.combobox_ids.grid(sticky=E, padx=padding + 3, pady=padding + 3)
        self.combobox_ids.bind("<<ComboboxSelected>>", self.combobox_users_callback)



    def combobox_users_callback(self, event):
        print(event.widget.get()," selected")
        self.index_user = event.widget.current()
        self.initialize_user_window()
        self.update_view(padding=5, ini_size=200)


    def update_view(self, padding = 5, ini_size = 200):
        self.center_label.config(text="Imagenes en las que ESTA " + self.user_ids[self.index_user]+" -- Page ("+str(self.current_pack_index)+")")
        self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=ini_size, padding=padding)

    def greet(self):
        print("Greetings!")

    def save_id(self):
        print("Saving ", self.user_ids[self.index_user], " data ...")
        cuestionario_df = pd.DataFrame([[self.annotator_id, self.user_ids[self.index_user], self.TP, self.FP, self.TN, self.FN]],
                                                              columns=["ID_anotador", "Nombre_personaje", "TP", "FP",
                                                                       "TN", "FN"])
        output_path = os.path.join(self.output_dir,
                                  self.name_form + "_" + self.annotator_id + "_" + "{0:0=5d}".format(self.index_user) + ".csv")
        cuestionario_df.to_csv(output_path, index=False, sep=";", header=True)

    def close_button_callback(self):
        #self.save_id()
        self.master.quit()

    def initialize_user_window(self):
        self.current_pack_index = 0
        self.in_user_imgs = True

        #path_clts = os.path.join(self.original_path_clt, self.user_ids[self.index_user].replace("_", " "))
        self.packs_of_imgs_user, self.packs_of_imgs_noise = self.create_imgs_packs(self.original_path_imgs,
                                                                   self.user_ids[self.index_user],group_size=self.group_size2show)
        self.max_packs_user = len(self.packs_of_imgs_user)
        self.max_packs_noise = len(self.packs_of_imgs_noise)
        self.TP = list(set([item for t in self.packs_of_imgs_user for item in t]).difference(set([None])))
        self.FP = []
        self.TN = list(set([item for t in self.packs_of_imgs_noise for item in t]).difference(set([None])))
        self.FN = []

    def next_id_button_callback(self):
        #self.save_id()
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
                text="Imagenes en las que ESTA " + self.user_ids[self.index_user] + " -- Page (" + str(
                    self.current_pack_index) + ")")
        else: #NOISE
            if (self.current_pack_index-1<0):
                self.in_user_imgs = True
                self.current_pack_index = self.max_packs_user-1
                self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(text="Imagenes en las que ESTA " + self.user_ids[self.index_user] +" -- Page ("+str(self.current_pack_index)+")")
            else:
                self.current_pack_index-=1
                self.fill_in_left_frame(self.packs_of_imgs_noise[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(
                    text="Imagenes en las que NO ESTA " + self.user_ids[self.index_user] + " -- Page (" + str(
                        self.current_pack_index) + ")")

    def rigth_mov_error_button_callback(self):
        if(self.in_user_imgs):
            if (self.current_pack_index+1 >= self.max_packs_user and len(self.packs_of_imgs_noise)>0):
                self.in_user_imgs = False
                self.current_pack_index = 0
                self.fill_in_left_frame(self.packs_of_imgs_noise[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(text="Imagenes en las que NO ESTA " + self.user_ids[self.index_user] +" -- Page ("+str(self.current_pack_index)+")")
            elif(self.current_pack_index+1 >= self.max_packs_user and len(self.packs_of_imgs_noise)<=0):return
            else:
                self.current_pack_index+=1
                self.fill_in_left_frame(self.packs_of_imgs_user[self.current_pack_index], ini_size=200, padding=5)
                self.center_label.config(
                    text="Imagenes en las que ESTA " + self.user_ids[self.index_user] + " -- Page (" + str(self.current_pack_index) + ")")
        else:
            if (self.current_pack_index + 1 >= self.max_packs_noise): return
            self.current_pack_index += 1
            self.fill_in_left_frame(self.packs_of_imgs_noise[self.current_pack_index], ini_size=200, padding=5)
            self.center_label.config(
                text="Imagenes en las que NO ESTA " + self.user_ids[self.index_user] + " -- Page (" + str(
                    self.current_pack_index) + ")")


    def fill_in_left_frame(self, img_paths, ini_size=200, padding = 5):
        n = 0
        self.map_button_img = []
        for i in range(1,4):
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
                    #btn.bind('<Button-1>', self.change_bg)
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
        print(event.widget["text"])



    def create_imgs_packs(self, path_imgs, user_id, group_size= 8):
        supposed_imgs_with_user = ast.literal_eval(self.complete_df.iloc[self.index_user]["TP"])+ast.literal_eval(self.complete_df.iloc[self.index_user]["FN"])
        supposed_imgs_without_user = ast.literal_eval(self.complete_df.iloc[self.index_user]["TN"])+ast.literal_eval(self.complete_df.iloc[self.index_user]["FP"])
        # Create complete list of imgs with path:
        list_imgs_user_clt = sorted(supposed_imgs_with_user)
        list_noise_user_clt = sorted(supposed_imgs_without_user)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-path-imgs", required=True,
        help="path to input directory of images")
    ap.add_argument("-o", "--output-path-answers", required=True,
        help="path to input directory of images")
    args = vars(ap.parse_args())
    input_path_imgs = args["input_path_imgs"]
    output_path_csv = args["output_path_answers"]


    #OPEN NEW APP WINDOW
    root = Tk()
    my_gui = validationGUI(root,input_path_imgs, output_path_csv)
    root.mainloop()
    #root.destroy()