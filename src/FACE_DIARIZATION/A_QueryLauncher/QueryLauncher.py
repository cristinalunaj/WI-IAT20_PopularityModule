import os
import PIL.Image as Image
import src.utils.files_utils as file_utils
import src.utils.loader as loader

class QueryLauncher():
    def __init__(self,participants_path, chrome_driver_path, logs_path, output_dir,
                 imgs2download=150,extra_info=None):
        self.participants = sorted(
            [item for sublist in loader.get_participants(participants_path) for item in sublist])
        self.logs_path = os.path.join(output_dir, "logs") if(logs_path=="") else logs_path
        self.output_dir = output_dir
        self.chrome_driver_path = chrome_driver_path
        self.imgs2download = imgs2download
        self.extra_info =extra_info
        self.current_participant = self.participants[0] + self.extra_info if self.extra_info != None else self.participants[0]

    def set_current_participant(self, participant):
        """
        Set current participant as participant+extra_info
            :param participant:(str) Participant name
        """
        self.current_participant = participant+self.extra_info if self.extra_info!=None else participant

    def clean_corrupted_imgs(self, accepted_formats = [".jpg", ".png", ".jpeg"]):
        """
        Remove those images that can't be accessed due to donwload errors
             :param accepted_formats: filter of imgs format accepted
           """
        imgs_path = os.path.join(self.output_dir, self.current_participant.replace(" ", "_"))
        for filename in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, filename)
            photo_name, extension = os.path.splitext(filename)
            if(not extension.lower() in accepted_formats):
                os.remove(img_path)
                continue
            try:
                _ = Image.open(img_path)
            except: #if problem opening images, then we consider that it's corrupt so we remove it
                os.remove(img_path)


    def rename_and_sort_imgs(self, img_directory=""):
        """
            Sort downloaded images by download name and change the name to: Participant_xxx.extension
            :param img_directory(str) : subdirectory where the images are (if there is any)
            """
        participant_imgs_path = os.path.join(self.output_dir, self.current_participant.replace(" ","_"), img_directory)
        #Rename images
        photos = os.listdir(participant_imgs_path)
        sorted_photos = file_utils.natural_string_sort(photos) #sort by download name
        index = 1
        for photo in sorted_photos:
            photo_name, extension = os.path.splitext(photo)
            original_path = os.path.join(participant_imgs_path, photo)
            new_path = os.path.join(participant_imgs_path, self.current_participant.replace(" ","_") + "_"+str("{0:0=3d}".format(index)) + extension)
            if(not os.path.exists(new_path)):
                os.rename(original_path,new_path)
            index += 1

    def download_imgs_from_keywords(self,**kwargs):
        """
            Query with the keywords and download the images that match the keyword
            :param img_directory(str) : subdirectory where the images are (if there is any)
            """
        raise NotImplementedError('Function "download_imgs_from_keywords" not implemented')
