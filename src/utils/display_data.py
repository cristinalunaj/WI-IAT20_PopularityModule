import os
from imutils import build_montages
from PIL import Image
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from random import randint
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.graph_objects as go

def get_mosaic(path_folder, user_id):
	if (not os.path.isdir(path_folder)):
		return
	imgs_in_folder = os.listdir(path_folder)
	imgs_2_show = []
	for img in imgs_in_folder:
		path_imgs = os.path.join(path_folder, img)
		face = Image.open(path_imgs)
		face.resize((96, 96))
		face_array = np.asarray(face)
		imgs_2_show.append(face_array)

	# create a montage using 96x96 "tiles" with 5 rows and 5 columns
	n_imgs = int(np.sqrt(len(imgs_2_show))) + 1
	mosaico = build_montages(imgs_2_show, (96, 96), (n_imgs, n_imgs))[0]
	mosaico = cv2.cvtColor(mosaico, cv2.COLOR_BGR2RGB)

	# show the output montage
	#ruta_etiqueta = output_path + query_choice
	title = "Cara ID:  #{}".format(user_id)
	cv2.imshow(title, mosaico)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_mosaic_plotly(path_folder, user_id):
	fig = go.Figure()

def get_n_of_faces_detected(root_path_bbox):
	folder_ids = os.listdir(root_path_bbox)
	total_faces_detected = 0
	for id in folder_ids:
		path_emb_id = os.path.join(root_path_bbox, id)
		data_embs = np.load(path_emb_id, allow_pickle=True)
		embs, embs_labels = data_embs['arr_0'], data_embs['arr_1']
		total_faces_detected+=embs.shape[0]
	print("Total faces detected: ", str(total_faces_detected))



def get_PCA_TSNE(labels,embeddings_caras_totales, output_path, plot2generate="pca"):
	print("Generating plots of clusters with PCA and TSNE....")
	embeddings_caras_totales = np.asarray(embeddings_caras_totales)
	etiquetaIDs = np.unique(labels)
	numCarasUnicas = len(np.where(etiquetaIDs > -1)[0])
	labels_con_ruido = pd.DataFrame()
	embeddings_con_ruido = pd.DataFrame()
	# Quitamos las muestras de ruido
	colors_con_ruido = {}

	for current_label in etiquetaIDs:
		index_2_find = np.where(labels == current_label)[0]
		if (current_label == -1):
			colors_con_ruido[current_label] = '#000000' #black for -1 label
			labels_con_ruido = labels_con_ruido.append(pd.DataFrame(labels[index_2_find]))
			embeddings_con_ruido = embeddings_con_ruido.append(pd.DataFrame(embeddings_caras_totales[index_2_find]))
		else:
			color = randint(0, 0xFFFFFF)
			colors_con_ruido[current_label] = '#%06X' % color
			labels_con_ruido = labels_con_ruido.append(pd.DataFrame(labels[index_2_find]))
			embeddings_con_ruido = embeddings_con_ruido.append(pd.DataFrame(embeddings_caras_totales[index_2_find]))

	labels_con_ruido =  np.reshape(labels_con_ruido._values,-1)
	embeddings_con_ruido = embeddings_con_ruido._values

	vectorizer_con_ruido = np.vectorize(lambda x: colors_con_ruido[x])

	if(plot2generate=="pca" or plot2generate=="both"):
		#CREATE PCA
		pca = PCA(n_components=3)
		pca.fit(embeddings_con_ruido)
		T_con = pca.transform(embeddings_con_ruido)


		#PLOTTING NOISE - 3D
		fig = plt.figure()
		ax = Axes3D(fig)
		# Pintamos
		ax.scatter(T_con[:, 0], T_con[:, 1], T_con[:, 2], c=vectorizer_con_ruido(labels_con_ruido), alpha=0.5)
		plt.title(str(numCarasUnicas) + ' clusters with PCA')
		plt.grid(True, which='both')
		plt.minorticks_on()
		plt.savefig(os.path.join(output_path,'clusters_pca_PlottingNoise.png'))
		plt.close(fig)

		#WITHOUT PLOTTING NOISE - 3D
		fig = plt.figure()
		ax = Axes3D(fig)
		index_2_find = np.where(labels_con_ruido > -1)[0] #remove noise data
		T_new_sin= T_con[index_2_find]
		labels_sin_ruido_new = labels_con_ruido[index_2_find]
		ax.scatter(T_new_sin[:, 0], T_new_sin[:, 1], T_new_sin[:, 2], c=vectorizer_con_ruido(labels_sin_ruido_new), alpha=0.5)
		plt.title(str(numCarasUnicas) + ' clusters with PCA')
		plt.grid(True, which='both')
		plt.minorticks_on()
		plt.savefig(os.path.join(output_path, 'clusters_pca_NotPlottingNoise.png'))
		plt.close(fig)
	if (plot2generate == "tsne" or plot2generate == "both"):
		# # Convertimos 128D a 3D mediante TSNE
		tsne = TSNE(n_components=3)
		T = tsne.fit_transform(embeddings_con_ruido)
		# Plot with noise - TSNE 3D
		fig = plt.figure()
		ax = Axes3D(fig)
		# #Pintamos
		ax.scatter(T[:, 0], T[:, 1], T[:, 2], c=vectorizer_con_ruido(labels_con_ruido),alpha=0.5)
		plt.title(str(numCarasUnicas) + ' clusters with TSNE')
		plt.grid(True, which='both')
		plt.minorticks_on()
		plt.savefig(os.path.join(output_path, 'clusters_tsne_PlotingNoise.png'))
		plt.close(fig)
		#withouth ploting noise - TSNE 3D
		fig = plt.figure()
		ax = Axes3D(fig)
		index_2_find = np.where(labels_con_ruido > -1)[0]  # remove noise data
		T_new_sin = T[index_2_find]
		labels_sin_ruido_new = labels_con_ruido[index_2_find]
		# #Pintamos
		ax.scatter(T_new_sin[:, 0], T_new_sin[:, 1], T_new_sin[:, 2], c=vectorizer_con_ruido(labels_sin_ruido_new), alpha=0.5)
		plt.title(str(numCarasUnicas) + ' clusters with TSNE')
		plt.grid(True, which='both')
		plt.minorticks_on()
		plt.savefig(os.path.join(output_path,'clusters_tsne_NotPlotingNoise.png'))
		plt.close(fig)

		print("Do plots of clusters with PCA and TSNE....")

# root_path_bbox = "/mnt/ESITUR2/TFG_VictorLoureiro/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN/boundingboxes_sum"
# get_n_of_faces_detected(root_path_bbox)
