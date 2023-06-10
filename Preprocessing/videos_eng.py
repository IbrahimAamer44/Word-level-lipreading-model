# importing libraries
import cv2
from lips import crop_lips
import mediapipe as mp
import numpy as np
from moviepy.editor import *
from align import read_align_eng
#from Dataset_eng import get_classes_indexes

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

"""
    IMPORTANT VARIABLES: 
                1. SEQUENCE_LENGTH variable in both word level and sentence level video function
                   [SEQUENCE_LENGTH is a constant number which specifies the limit of frames per sequence,
                   SKIP_FRAME_WINDOW is used to make sure each frame picked is evenly spread through the seq]
"""

"""
    TO-DO:
        1. Expecting an exception when an align is shorter than sequence length (word level)
            [Prevent from happening]
"""

"""
    This function does the following :
        1. converts a clip ('VideoFileClip' of MoviePie) to a list of frames
        2.It will also crop each frame over lips
"""
def clip_to_list(clip, SEQUENCE_LENGTH):

    # Getting frames of the clip
    frames = clip.iter_frames()

    # The list of frames which is to be returned
    frame_list = []

    # Using mediapipe model to detect face
    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        # Iterating over frames
        for i in frames:
            frame_list.append(crop_lips(i, holistic))

    # Get the total number of frames in the video.
    video_frames_count = int(len(frame_list))
    #print("v :", video_frames_count)

    # Calculate the interval after which frames will be added to the list.
    #skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    processed_frames_list = []


    if video_frames_count < SEQUENCE_LENGTH:
        # Padding the sequence with last frame of segment

        processed_frames_list = frame_list # Copying original frames

        # Padding with last frame
        for cnt in range(video_frames_count, SEQUENCE_LENGTH):
            processed_frames_list.append(frame_list[video_frames_count-1])


    else:
        # remove extra frames from the end
        processed_frames_list = frame_list[0:SEQUENCE_LENGTH] # Copying first SEQ_LEN frames of the segment


    # Normalizing frames
    normalized_frames = []
    for fr in processed_frames_list:

        normalized_frames.append(fr / 255)


    # Iterate through the Video Frames.
    #for frame_counter in range(SEQUENCE_LENGTH):

        # Getting the appropriate frame from original list and normalizing it
        #normalized_frame = frame_list[frame_counter * skip_frames_window] / 255

        # Appending the normalized frame in new list
        #processed_frames_list.append(normalized_frame)

    return normalized_frames

"""
    Function is:
        1. Used for word level lip-reading
        2. And gets segments(words) from sentence in the video
"""
def read_and_segment_video(file_directory):

    # Specify the number of frames of a video that will be fed to the model as one sequence.
    SEQUENCE_LENGTH = 13

    clips_list = [] #data
    labels = []

    align = read_align_eng(file_directory)


    try:

        # loading video using MoviePie function
        main_video = VideoFileClip(file_directory)

        # Segmenting video using align of each word
        for ind in align:

            try:
                # Getting subclip
                subclip = clip_to_list(main_video.subclip(ind[1], ind[2]), SEQUENCE_LENGTH)

                # getting video for only starting 10 seconds
                clips_list.append(subclip)

                # Appending the word as label in the label_list
                labels.append(ind[0])

            except Exception as e:
                print(2)
                print(e)

    except Exception as e:
        print(1)
        print(e)


    return clips_list, labels

def Get_sentences(path):

    #path = '..\\Dictionary\\roman_urdu_sentences.txt'

    # Using readlines()
    file = open(path, 'r')
    lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace(" ", "_")
        lines[i] = lines[i].replace("\n", "")

    return lines

if __name__ == "__main__":

    """# loading video dsa gfg intro video
    clip = VideoFileClip("..\\Dataset\\Urdu\\2\\aap_kese_hai_jee_kyun_do\\_video.avi")

    #frames = clip_to_list(clip,7)
    """

    #print(len(frames))

    #sentences = Get_sentences('..\\Dictionary\\roman_urdu_sentences.txt') # Local function

    classes_list = ['sil', 'at', 'five', 'bin', 'red', 'two', 'a', 'j', 'green', 'p', 'eight', 'now', 'place', 'again', 'f', 'b', 'nine', 'n', 'o', 'lay', 'with', 'g', 'q', 's', 'x', 'in', 'd', 'four', 'soon', 'one', 'k', 'v', 'please', 'c', 'e', 'y', 'z', 'i', 'blue', 'by', 'zero', 'l', 'u', 'seven', 't', 'set', 'h', 'three', 'r', 'm', 'white', 'six']

    features = []
    labels = []

    path = "..\\Dataset\\GC\\s1"
    sentence_list = os.listdir(path)
    sentence_list.remove('align')
    """
    s_id = 1
    for cnt in range(1,4):
        path = "..\\Dataset\\GC\\s"+str(cnt)
        break
    dir_list = os.listdir(path)     
    """

    #for i in range(len(sentence_list)):
    for i in range(0,2):

        vid_path = path + "\\" + sentence_list[i]

        print(i, ". ", sentence_list[i])

        clips, lbl = read_and_segment_video(vid_path)

        features.extend(clips)

        # Getting index of current class
        classes_index = get_classes_indexes(lbl, classes_list)

        labels.extend(classes_index)

        # TEMP
        #if i == 4:
        #    break

    print(len(features))
    print(len(labels))

    #for i in range(len(clips)):
        #print(labels[i], " ", len(clips[i]))





