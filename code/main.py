from pipeline import pipeline, pipelineSingleFrameDetection, threshold, historyFactor

from moviepy.editor import VideoFileClip

output = '../processed_project_video_h%1.2f_t%1.2f.mp4' % (historyFactor, threshold)
clip = VideoFileClip("../project_video.mp4")
input_clip = clip.fl_image(pipeline)
input_clip.write_videofile(output, audio=False)