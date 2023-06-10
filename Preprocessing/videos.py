# importing libraries
import cv2
from lips import crop_lips
import mediapipe as mp
import numpy as np
from moviepy.editor import *
from align import read_align
#from Dataset import get_sentences

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





    video_frames_count = int(len(frame_list))
    if video_frames_count < SEQUENCE_LENGTH:
        # Padding the sequence with last frame of segment

        processed_frames_list = frame_list # Copying original frames

        # Padding with last frame
        for cnt in range(video_frames_count, SEQUENCE_LENGTH):
            processed_frames_list.append(frame_list[video_frames_count-1])


    if len(frame_list) > SEQUENCE_LENGTH:
        # remove extra frames from the end
        processed_frames_list = frame_list[0:SEQUENCE_LENGTH] # Copying first SEQ_LEN frames of the segment
    else:
        processed_frames_list = frame_list

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
        [args = file_id, augment flag to augment the video if set true(default false)]
"""

def read_and_segment_video(file_directory, sq_len, augment=False):

    # Specify the number of frames of a video that will be fed to the model as one sequence.
    SEQUENCE_LENGTH = sq_len

    clips_list = [] #data
    labels = []
    input_len = [] # List to hold each features input length(no. of frames)

    align = read_align(file_directory)

    try:

        # loading video using MoviePie function
        main_video = VideoFileClip(file_directory + "\\_video.avi")

        # Augmenting the original video if flag true
        if augment == True:
            # Mirroring the original video for DATA AUGMENTATION
            main_video = main_video.fx(vfx.mirror_x)

        # Segmenting video using align of each word
        for ind in align:

            # Not considering 'س'[sil] segment
            if ind[0] != 'س':

                try:
                    segment = main_video.subclip(ind[1], ind[2])

                    frames = segment.iter_frames()
                    # counting frames
                    cnt = 0
                    for i in frames:
                        cnt += 1
                    #print("Num frames = ", cnt)

                    # Segment will be ignored if it's shorter than 6 frames
                    if cnt > 6:

                        # Getting subclip
                        subclip = clip_to_list((segment), SEQUENCE_LENGTH)


                        # getting video for only starting 10 seconds
                        clips_list.append(subclip)

                        # Appending the word as label in the label_list
                        labels.append(ind[0])


                        input_len.append(cnt)


                except Exception as e:
                    print(e)

    except Exception as e:
        print(e)


    return clips_list, labels, input_len


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

    SEQUENCE_LENGTH = 18

    sentences = Get_sentences('..\\Dictionary\\roman_urdu_sentences.txt') # Local function

    features = []
    labels = []
    input_len = []

    for i in range(8,len(sentences)):
        path = "..\\Dataset\\Urdu\\0\\" + sentences[i]


        # Getting original features,labels
        clips, lbl, len1 = read_and_segment_video(path, SEQUENCE_LENGTH, augment=False)

        # Getting augmented features,labels
        clips_aug, lbl_aug, len2 = read_and_segment_video(path, SEQUENCE_LENGTH, augment=True)

        #Appending both features and labels to the same array
        features.extend(clips)
        labels.extend(lbl)
        input_len.extend(len1)

        features.extend(clips_aug)
        labels.extend(lbl_aug)
        input_len.extend(len2)


        break

    print(len(features))
    print(len(labels))
    print(len(input_len))

    print("Features len - Input len")
    for i in range(len(features)):
        print(len(features[i]), "  ", input_len[i])

'''
    print(" - ",input_len[1])
    for i in range(0,input_len[1]):
        vis = np.concatenate((features[1][i], features[7][i]), axis=1)

        cv2.imshow("Image", vis)

        cv2.waitKey(0)
'''




