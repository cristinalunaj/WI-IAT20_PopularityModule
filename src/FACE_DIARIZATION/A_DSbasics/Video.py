import cv2, os
import subprocess
"""
    Video class that represents a video object
	author: Cristina Luna, Ricardo Kleinlein
	date: 05/2020
"""
class Video():
    def __init__(self, video_name, input_video_folder):
        self.video_name = video_name
        self.input_video_folder = input_video_folder
        self.duration = self.get_duration()
        self.width,self.heigth = self.get_size()
        self.fps = self.get_fps()

    def get_duration(self):
        """
        Get video duration in seconds
        :return: (float) video duration in secs.
        """
        duration_output = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             os.path.join(self.input_video_folder, self.video_name)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if("Invalid frame dimension" in duration_output.stdout.decode()):
            return float(duration_output.stdout.decode().split("\n")[-2])
        else:
            return float(duration_output.stdout)

    def get_size(self):
        """
        Get frames shape (width, heigth)
        :return: (int, int) width, heigth
        """
        video = cv2.VideoCapture(os.path.join(self.input_video_folder, self.video_name))
        return int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self):
        """
        Get frames per second
        :return: (float) fps
        """
        video = cv2.VideoCapture(os.path.join(self.input_video_folder, self.video_name))
        return video.get(cv2.CAP_PROP_FPS)

