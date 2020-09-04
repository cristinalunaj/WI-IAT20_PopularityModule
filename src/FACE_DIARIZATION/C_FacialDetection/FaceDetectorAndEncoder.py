
"""
	Face detection and encoding
	Our approach is based on visual information.
	Run through the frames of a program, and detect as many faces as
	possible using MTCNN [1] or HaarCascade [2]
	For each detected face, encode its features as a vector
	embedding, thanks to the Facenet model [3].
	That way, each face, no matter from whom, available in a broadcast
	will be accesible as a rich latent representation.
	[1]: https://github.com/ipazc/mtcnn (code) // https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf (paper)
	[2]: https://github.com/opencv/opencv/tree/master/data/haarcascades
	[3]: https://arxiv.org/abs/1503.03832 (paper)
	author: Cristina Luna, Ricardo Kleinlein
	date: 05/2020
	Usage:
		python3 FaceDetectorAndEncoder.py
        --face-detector MTCNN
        --root-input-folder ~/RTVE2018DB/test/GENERATED_ME/DATASET_GOOGLE_IMGS/refactor_DS
        --program-name None
        --output-dir ~/RTVE2018DB/test/GENERATED_ME/DATASET_GOOGLE_IMGS/VIDEO_DB_MTCNN
        --program-participants-folder ~/RTVE2018DB/test/rttm_INFO/FACEREF/participants
        --imgs-2-maintain 100
        --face-threshold 0.98
        --extract-single-face False
	Options:
	    --face-detector: Face detector (Options: MTCNN or HaarCascade)
		--encoding-model: Path to the encoding model pretrained weigths
		[default: ../../../data/models/pre_trained_models/face_embs_model/facenet_keras.h5]
		--input-frames-folder: Path to folder with the frames of the videos
		--video-name: Name of the target video folder (if None, then process all videos in folder 'input_imgs_folder')
		--output-dir: Directory to save results in
		--imgs-2-maintain: Number of images per participant/program to maintain. [default: All]
        --face-threshold: Probability[0-1] of accept a face as correct . Those faces below the face_threshold will not be considered in what follows.
        --extract-single-face: True if we want to extract a single face from frame (This face will be the biggest one in the image)
		--quiet	Hide visual information
		-h, --help	Display script additional help
"""

import os, time
from PIL import Image
import numpy as np
from numpy import savez_compressed
from keras.models import load_model
import cv2
from mtcnn.mtcnn import MTCNN
from src.BaseArgs import FaceDetEncArgs
import src.utils.loader as loader

default_path_HaarCascade = "../../../data/models/pre_trained_models/face_detectors/haarcascade_frontalface_default.xml"



class FaceDetectorAndEncoder():
    def __init__(self, face_detector, encoding_model, root_input_folder, path_participants_folder,
                 output_dir, program_name, imgs_after_filtering=100):
        self.root_input_folder = root_input_folder
        self.programs = [file.split(".")[0] for file in os.listdir(path_participants_folder)] \
            if(program_name == None or program_name == 'None') else [program_name]
        self.output_dir = output_dir
        self.imgs_after_filtering = imgs_after_filtering
        self.flag_face_detector = 0
        self.face_detector = self._load_face_detection_model(face_detector)  # MTCNN
        self.encoding_model = load_model(encoding_model, compile=False)  # FaceNet


    def _load_face_detection_model(self, detector="MTCNN"):
        """
        Load face detector. Two possible options: MTCNN or HaarCascade
            :param detector: flag with the name of the detector to use
            :return: loaded model detector
        """
        if(detector=="MTCNN"):
            self.flag_face_detector = "MTCNN"
            return MTCNN()
        elif(detector=="HaarCascade"):
            #HAARCASCADE -> haarcascade_frontalface_default.xml
            self.flag_face_detector = "HaarCascade"
            return cv2.CascadeClassifier(default_path_HaarCascade)

    def get_embedding(self, face_pixels):
        """
        Get embeddings from face image pixels using FaceNet model
        Embeddings extracted from FaceNet that represents the face in a 128-dimensional-latent-space
            :param face_pixels: (array) Value of faces pixels detected
            :return: encoded face as embedding
       """
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.encoding_model.predict(samples)
        return yhat[0]

    def detect_face_MTCNN(self,rgb_image):
        """
        Detect faces with MTCNN
            :param rgb_image: image in RGB
            :return: detected faces as dict with MTCNN detection information [bbox, confidence...]
        """
        return self.face_detector.detect_faces(rgb_image)

    def detect_face_HaarCascade(self,frame_path):
        """
        Detect faces with HaarCascade
            :param frame_path: image in grayscale
            :return: detected faces as dict with HaarCascade detection information reported as MTCNN [bbox]
        """
        gray_image = loader.load_image(frame_path, colormode="grayscale")
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')
        faces = self.face_detector.detectMultiScale(gray_image, 1.3, 5)
        faces_as_dict = [{'box': f,
                          'confidence': 1.0} for f in faces]
        return faces_as_dict


    def create_output_folders(self, program,ID, sub_folder_extra_name="", create_face_folder=True):
        """
        Create output sub-folders to save the embeddings generated, bbox, face images and the dicts generated
        by MTCNN (mtcnn_debug saves info about faces size,position,confidence...)
        :param ID: name of the folder with the ID frames
        :return: output folders path
        """
        if(create_face_folder):
            output_path_face = os.path.join(self.output_dir, program, "faces"+sub_folder_extra_name,ID)
            os.makedirs(output_path_face, exist_ok=True)
        else:
            output_path_face = ""
        output_path_bbox = os.path.join(self.output_dir,program, "boundingboxes"+sub_folder_extra_name, ID)
        output_path_emb = os.path.join(self.output_dir,program, "embeddings"+sub_folder_extra_name, ID)
        output_path_mtcnn = os.path.join(self.output_dir,program, "mtcnn_debug"+sub_folder_extra_name,ID)
        if (not os.path.exists(output_path_bbox)):
            os.makedirs(output_path_bbox)
            os.makedirs(output_path_mtcnn)
            os.makedirs(output_path_emb)
        return output_path_face,output_path_bbox, output_path_emb, output_path_mtcnn

    def extract_biggest_face(self, detected_faces, confidence_th):
        """
        Extract the biggest face in detected_faces
        :param detected_faces: List of detected faces by MTCNN or HaarCascade
        :return: AList with single element, the biggest face
        """
        single_face_dict = {}
        max_size = 0
        for face in detected_faces:
            if(face["confidence"]>=confidence_th):
                _, _, width, height = face['box']
                face_size = width*height
                if(face_size>max_size):
                    max_size = face_size
                    single_face_dict = face
        return [single_face_dict]


    def detect_and_encode_faces(self,required_size = (160, 160), extract_single_face=False, probability_of_being_face_accepted=0.0):
        """
           Get embeddings and bounding boxes from images. The function will save the bounding box of the faces detected, their embeddings after passing through FaceNet,
           the metadata generated by MTCNN and their cutted faces
                :param required_size: Size that the face encoder (FaceNet) expects as input
       """
        for program in self.programs:
            input_path_program = os.path.join(self.root_input_folder, program)
            for id in sorted(os.listdir(input_path_program)):
                if (os.path.isdir(os.path.join(input_path_program, id)) and
                        not os.path.exists(os.path.join(self.output_dir,program, "boundingboxes", id))):
                    print("Extracting faces & bbox from program: ", program, " - id: ", id)
                    #Define and create ROOT output dirs
                    output_path_video,output_path_bbox, output_path_emb, output_path_mtcnn = self.create_output_folders(program,id)
                    # path of identities/programs with their iamges or frames
                    path_frames = os.path.join(input_path_program, id)
                    # frames or images
                    for frame in sorted(os.listdir(path_frames)):
                        print("processing ", frame)
                        path_frame = os.path.join(input_path_program, id, frame)
                        frame_name = frame.split(".")[0]
                        try:
                            X, y, mtcnn_info, emb_data = list(), list(), list(), list()
                            face_queries,labels,mtcnn_metadata,total_embeddings = list(), list(), list(), list()
                            # load image from file, convert to RGB and convert to array
                            rgb_img = np.asarray(Image.open(path_frame).convert('RGB'))
                            #get bounding boxes using MTCNN
                            detected_faces = self.detect_face_MTCNN(rgb_img) if(self.flag_face_detector=="MTCNN") \
                                else self.detect_face_HaarCascade(path_frame)

                            #Extract single face (the biggest one) - e.g. for OCR Images
                            if(extract_single_face):
                                detected_faces = self.extract_biggest_face(detected_faces, probability_of_being_face_accepted)


                            # Extract the bounding box of all the faces detected on the photo
                            if (len(detected_faces) != 0):
                                for n in range(len(detected_faces)):
                                    x1, y1, width, height = detected_faces[n]['box']
                                    # Fix a possible bug
                                    x1, y1 = abs(x1), abs(y1)
                                    x2, y2 = x1 + width, y1 + height

                                    #Convert image into a numpy array
                                    image = Image.fromarray(rgb_img[y1:y2, x1:x2])
                                    resized_img = image.resize(required_size)
                                    #save_img
                                    resized_img.save(os.path.join(output_path_video, frame_name+"_"+str(n)+".png"))
                                    # Change dimension of face to FaceNet input image size (160x160)
                                    face_array = np.asarray(resized_img)
                                    #get embeddings using FaceNet
                                    emb_data.append(self.get_embedding(face_array))
                                    face_queries.append(face_array)
                                    labels.append(id+"/"+frame)
                                    mtcnn_metadata.append(detected_faces[n])

                                # Save results
                                X.extend(face_queries)
                                y.extend(labels)
                                mtcnn_info.extend(mtcnn_metadata)
                                total_embeddings.extend(emb_data)

                                savez_compressed(os.path.join(output_path_bbox, frame_name + ".npz"), np.asarray(X), np.asarray(y))
                                savez_compressed(os.path.join(output_path_mtcnn, frame_name + ".npz"), np.asarray(mtcnn_info), np.asarray(y))
                                savez_compressed(os.path.join(output_path_emb, frame_name + ".npz"), np.asarray(total_embeddings), np.asarray(y))
                        except:
                            print('ERROR DETECTED IN : {:s}'.format(path_frame))

    def filter_bbox_embs(self,to_filter="embeddings",probability_of_being_face_accepted = 0.98, face_size_accepted = 80*80):
        """
        Filter the number of images/embs/bboxes to use to the number indicated in imgs_2_maintain
           Args:
            root_path (str): Input root path of the extracted embeddings/bboxes
            output_path (str): Path in which we will create the new folders with the extraced information (bbox/faces/embeddings...) of the reduced number of images
            input_path_MTCNN_debug (str): Input root path of the mtcnn_metadata
            imgs_2_maintain (int): Number of photos to maintain from those donwloaded
            probability_of_being_face_accepted (double): Probability extraxted by MTCNN of being accepted as face
            face_size_accepted (int): Minimum size of face accpeted (in pixels)
        """
        for program in self.programs:
            input_path_non_filtered_embs = os.path.join(self.output_dir, program, to_filter)
            output_path_filtered = os.path.join(self.output_dir, program,to_filter + "_" + str(self.imgs_after_filtering))
            for id in sorted(os.listdir(input_path_non_filtered_embs)):
                print("Filtering program: ", program, " - id: ", id)
                if (os.path.exists(os.path.join(output_path_filtered, id))):
                    continue
                os.makedirs(os.path.join(output_path_filtered, id), exist_ok=True)
                counter_imgs_copied = 0
                list_of_embs = sorted(os.listdir(os.path.join(input_path_non_filtered_embs, id)))
                for emb in list_of_embs:
                    embs_final = []
                    labels_final = []
                    # embs
                    path_embs_output = os.path.join(output_path_filtered, id, emb)
                    path_embs_input = os.path.join(input_path_non_filtered_embs, id, emb)
                    data_embs = np.load(path_embs_input, allow_pickle=True)
                    embs_info, embs_labels = data_embs['arr_0'], data_embs['arr_1']

                    # mtcnn
                    path_mtcnn_debug_input = os.path.join(self.output_dir, program,"mtcnn_debug", id, emb)
                    data_mtcnn = np.load(path_mtcnn_debug_input, allow_pickle=True)
                    mtcnn_info, mtcnn_labels = data_mtcnn['arr_0'], data_mtcnn['arr_1']

                    for sub_face_index in range(len(mtcnn_info)):
                        _, _, width, height = mtcnn_info[sub_face_index]['box']
                        face_size = width * height
                        probability_of_being_face = mtcnn_info[sub_face_index]["confidence"]
                        if (face_size < face_size_accepted or probability_of_being_face < probability_of_being_face_accepted):
                            continue
                        else:
                            embs_final.append(embs_info[sub_face_index])
                            labels_final.append(embs_labels[sub_face_index])
                    # image 2 copy
                    if (counter_imgs_copied < self.imgs_after_filtering and len(embs_final) >= 1):
                        # shutil.copy(path_embs_input, path_embs_output)
                        savez_compressed(path_embs_output, np.asarray(embs_final), np.asarray(labels_final))
                    else:
                        continue
                    counter_imgs_copied += 1

    def generate_compact_npz(self,to_filter="embeddings", replace_by_spaces=False):
        """
           Compact the inidvidual embs/bbox in a single compressed file (.npz) per participant/program
                   Args:
                    input_path (str): Input root path of the extracted embeddings/bbox organised in folders per user
                    output_path (str): Path in which we will save the new compressed embeddings/bboxes
           """
        for program in self.programs:
            input_path = os.path.join(self.output_dir, program,to_filter + "_" + str(self.imgs_after_filtering))
            output_path = os.path.join(self.output_dir, program, to_filter+"_sum")
            for id in sorted(os.listdir(input_path)):
                input_id_embs = os.path.join(input_path, id)
                list_of_embs = sorted(os.listdir(input_id_embs))
                embs_totales_query = list()
                etiquetas_embs_total = list()
                for emb in list_of_embs:
                    ruta_embedding = os.path.join(input_id_embs, emb)
                    data_emb = np.load(ruta_embedding, allow_pickle=True)
                    emb_info, etiquetas_emb = data_emb['arr_0'], data_emb['arr_1']
                    for value in emb_info:
                        embs_totales_query.append(value)
                    for value in etiquetas_emb:
                        etiquetas_embs_total.append(value)
                embs_final = np.asarray(embs_totales_query)
                embs_label = np.asarray(etiquetas_embs_total)
                if (len(embs_final) >= 1):
                    os.makedirs(output_path, exist_ok=True)
                    if (replace_by_spaces):
                        new_id_name = id.replace("_", " ")
                    else:
                        new_id_name = id
                    np.savez_compressed(output_path + '/' + new_id_name, embs_final, embs_label)

if __name__ == "__main__":

    face_detEnc_args_obj = FaceDetEncArgs()
    args = face_detEnc_args_obj.parse()

    if(args.imgs_2_maintain <=0):
        imgs_after_filtering = 2000000
    else:
        imgs_after_filtering = args.imgs_2_maintain

    face_det_enc_obj = FaceDetectorAndEncoder(args.face_detector, args.encoding_model, args.root_input_folder,
                                              args.program_participants_folder, args.output_dir,
                                              args.program_name, imgs_after_filtering = imgs_after_filtering)
    #MTCNN & FaceNet
    face_det_enc_obj.detect_and_encode_faces(extract_single_face=args.extract_single_face,probability_of_being_face_accepted = args.face_threshold)
    #Filter results
    face_det_enc_obj.filter_bbox_embs(to_filter="embeddings",probability_of_being_face_accepted = args.face_threshold,
                                             face_size_accepted = 80*80)
    face_det_enc_obj.filter_bbox_embs(to_filter="boundingboxes", probability_of_being_face_accepted= args.face_threshold,
                                             face_size_accepted= 80 * 80)
    face_det_enc_obj.filter_bbox_embs(to_filter="mtcnn_debug", probability_of_being_face_accepted= args.face_threshold,
                                             face_size_accepted= 80 * 80)

    #Generate compact version of filtered data:
    time.sleep(3 * 60)  # 3 min -> wait to let filter finish the copy/paste/modification of the data
    replace_by_spaces = True #True en LN24H False en L6N (para Google y program, True en OCR)
    face_det_enc_obj.generate_compact_npz(to_filter="embeddings", replace_by_spaces=replace_by_spaces)
    face_det_enc_obj.generate_compact_npz(to_filter="boundingboxes", replace_by_spaces=replace_by_spaces)
    face_det_enc_obj.generate_compact_npz(to_filter="mtcnn_debug", replace_by_spaces=replace_by_spaces)











