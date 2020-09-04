"""
	Downsample videos as a sequence of frames
	Broadcast videos usually have a high rate of frames per second (FPS).
	Working on them all would result in a high computational expense,
	and long processing times.
	As a trade-off between performance and expense, we split the original
	recorded audiovisual material in frames. In other words,
	we downsample the original programs.
	author: Ricardo Kleinlein, Cristina Luna
	date: 05/2020
	Usage:
		(e.g. python3 FrameExtractor.py
		--input-videos-folder ~/videos
		--video-name None
		--fps 0
		--frame-height 0
		--frame-width 0
		--output-dir ~/prueba
		--extraction-lib ffmpeg
		--extract-parallel-video-frames True)
	Options:
		--input-videos-folder Path to videos folder (folder with the video/s to extract frames)
		--video-name Name of the video to analyse (if we want to extract a single video, else None)
		--fps  Set the FPS ratio of the outcome (if 0, then extract default fps of video)
		--frame-height	Set the height of the frames
		--frame-width	Set the width of the frames
		--extract-parallel-video-frames True if we want to use parallel processing to extract frames of all the videos in input_videos_folder
		--output-dir	Directory to save results in
		--extraction-lib Extract frames with opencv or with ffmpeg (options: ffmpeg or opencv)
		--quiet	Hide visual information
		-h, --help	Display script additional help
"""

import os, time, multiprocessing
from src.BaseArgs import FrameExtractorArgs
import cv2
from src.FACE_DIARIZATION.A_DSbasics.Video import Video

def unwrap_self_extract_frames_ffmpeg(arg, **kwarg):
	"""
	Function necessary for doing parallel processing using ffmpeg
	:return: objective function for frames extraction using ffmpeg
	"""
	return FramesExtractor.extract_frames_ffmpeg(*arg, **kwarg)

def unwrap_self_extract_frames_opencv(arg, **kwarg):
	"""
	Function necessary for doing parallel processing using opencv
	:return: objective function for frames extraction using opencv
	"""
	return FramesExtractor.extract_frames_opencv(*arg, **kwarg)

class FramesExtractor():
	def __init__(self, input_videos_folder, video_name, new_fps, new_frame_width, new_frame_height,
				 output_dir, extraction_lib, parallel_processing=False):
		self.input_videos_folder = input_videos_folder
		self.fps = new_fps
		self.frame_width = new_frame_width
		self.frame_height = new_frame_height
		self.output_dir = output_dir
		self.extraction_lib = extraction_lib
		self.parallel_processing = parallel_processing
		self.videos = sorted(os.listdir(self.input_videos_folder)) if (
					video_name == None or video_name == 'None') else [video_name]


	def extract_frames_opencv(self,video_name, extension=".png"):
		"""
		Extract frames from video using opencv
		:param video_name: name of video to extract frames from
		:param extension: output images extension
		"""
		print("Start frames extraction of: ", video_name)
		video_obj = Video(video_name, self.input_videos_folder)
		video_name_wout_extension = video_name.split(".")[0]
		new_dir = os.path.join(self.output_dir,'frames', video_name_wout_extension)
		if (not os.path.isdir(new_dir)):
			os.makedirs(new_dir)
			vidcap = cv2.VideoCapture(os.path.join(self.input_videos_folder, video_name))
			fps_original = video_obj.fps
			if (self.fps == 0):
				# extract all frames
				self.fps = fps_original
			check_fps = int(fps_original / self.fps)
			success, image = vidcap.read()
			count = 0
			index_name = 1
			while success:
				success, image = vidcap.read()
				if (count % check_fps == 0):
					new_name = video_name_wout_extension + "_" + "{0:0=6d}".format(index_name)
					resized_img = self.get_resized_image(image, self.frame_width, self.frame_height)
					img_output_path = os.path.join(new_dir, new_name + extension)
					cv2.imwrite(img_output_path, resized_img)  # save frame as JPEG file
					index_name+=1
				count += 1

	def get_resized_image(self,image, img_width, img_heigth):
		"""
		Resize image from original shpae (w,h) to (img_width,img_heigth)
		:param image:
		:param img_width: new width
		:param img_heigth: new heigth
		:return: image with new shape
		"""
		if (img_heigth != 0 and img_width != 0):
			return cv2.resize(image, (img_width, img_heigth))
		else:
			return image

	def extract_frames_ffmpeg(self, video_name, extension=".png"):
		"""
		Extract frames from video using ffmpeg
		:param video_name: name of video to extract frames from
		:param extension: output images extension
		"""
		video_obj = Video(video_name, self.input_videos_folder)
		video_name_wout_extension = video_name.split(".")[0]
		if (self.frame_height == 0 or self.frame_width == 0):
			self.frame_height = video_obj.heigth
			self.frame_width = video_obj.width
		if(self.fps == 0):
			self.fps = int(video_obj.fps)
		if (not os.path.exists(os.path.join(self.output_dir, 'frames',video_name_wout_extension))):
			os.makedirs(os.path.join(self.output_dir, 'frames',video_name_wout_extension))
			cmd = 'ffmpeg -i ' + os.path.join(self.input_videos_folder,video_name) + ' '
			cmd += '-vf '
			cmd += 'fps=' + str(self.fps) + ' '
			cmd += '-s ' + str(self.frame_width)
			cmd += 'x' + str(self.frame_height) + ' '
			cmd += os.path.join(self.output_dir,'frames',video_name_wout_extension ,video_name_wout_extension + '_%06d'+extension)
			print(cmd)
			os.system(cmd)
			print('\n Program ' + video_name + ' already has its frame extracted\n')

	def extract_FPS_parallel(self):
		"""
		Extract frames in a parallel way using ffmpeg or opencv funcions
		"""
		start_time = time.time()
		pool = multiprocessing.Pool()  # processes = 7
		if(self.extraction_lib=="ffmpeg"):
			pool.map(unwrap_self_extract_frames_ffmpeg, zip([self] * len(self.videos), self.videos))
		else:
			pool.map(unwrap_self_extract_frames_opencv, zip([self] * len(self.videos), self.videos))
		pool.close()
		pool.join()
		final_time = (time.time() - start_time)
		print("--- %s Data preparation TIME IN min ---" % (final_time / 60))



if __name__ == "__main__":
	frame_extractor_args_obj = FrameExtractorArgs()
	args = frame_extractor_args_obj.parse()
	frames_extractor = FramesExtractor(args.input_videos_folder, args.video_name, args.fps, args.frame_width,
					   args.frame_height, args.output_dir, args.extraction_lib, args.parallel_processing)
	if(frames_extractor.parallel_processing):
		frames_extractor.extract_FPS_parallel()
	else:
		for video_name in frames_extractor.videos:
			if(frames_extractor.extraction_lib=="opencv"):
				frames_extractor.extract_frames_opencv(video_name)
			else:
				frames_extractor.extract_frames_ffmpeg(video_name)

