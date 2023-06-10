import os

from videos_eng import read_and_segment_video
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
"""
    TO-DO:
        FIX create_word_level_dataset(): Only gets videos of single speaker change so that
                                         it reads videos of all the speakers.
"""


def get_sentences(path):

    #path = '..\\Dictionary\\roman_urdu_sentences.txt'

    # Using readlines()
    file = open(path, 'r')
    lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace(" ", "_")
        lines[i] = lines[i].replace("\n", "")

    return lines

def create_word_level_dataset():

    # TEMP: change back to original !!!!
    #rom_classes_list = ['<sil>', 'ek', 'paanch', 'hum', 'han', 'chaar', 'kyun', 'wo', 'theen', 'kab', 'nai', 'tha', 'aap', 'nau', 'thay', 'kitne', 'hoon', 'do', 'jee', 'hai', 'kidhar', 'kon', 'konsa', 'mein', 'aath', 'kese', 'chhae', 'saath']
    classes_list = ['sil', 'at', 'five', 'bin', 'red', 'two', 'a', 'j', 'green', 'p', 'eight', 'now', 'place', 'again', 'f', 'b', 'nine', 'n', 'o', 'lay', 'with', 'g', 'q', 's', 'x', 'in', 'd', 'four', 'soon', 'one', 'k', 'v', 'please', 'c', 'e', 'y', 'z', 'i', 'blue', 'by', 'zero', 'l', 'u', 'seven', 't', 'set', 'h', 'three', 'r', 'm', 'white', 'six']

    features = []
    labels = []

    path = "..\\Dataset\\GC\\s1"
    sentence_list = os.listdir(path)
    sentence_list.remove('align')
    #sentence_list.remove('Thumbs.db')
    """
    s_id = 1
    for cnt in range(1,4):
        path = "..\\Dataset\\GC\\s"+str(cnt)
        break
    dir_list = os.listdir(path)     
    """

    for i in range(len(sentence_list)):

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


    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels


"""
    Function receives a list of labels e.g. ['<sil>', 'wo', 'kese', 'hoon', 'jee', 'kab', 'ek']
    and converts it into corresponding list containing index of each label in the list of all classes
"""
def get_classes_indexes(labels, classes_list):

    classes_indexes = []

    for lbl in labels:

        try:
            classes_indexes.append(classes_list.index(lbl))
        except:
            classes_indexes.append(0)

    return classes_indexes


if __name__ == "__main__":

    classes_list = ['sil', 'at', 'five', 'bin', 'red', 'two', 'a', 'j', 'green', 'p', 'eight', 'now', 'place', 'again', 'f', 'b', 'nine', 'n', 'o', 'lay', 'with', 'g', 'q', 's', 'x', 'in', 'd', 'four', 'soon', 'one', 'k', 'v', 'please', 'c', 'e', 'y', 'z', 'i', 'blue', 'by', 'zero', 'l', 'u', 'seven', 't', 'set', 'h', 'three', 'r', 'm', 'white', 'six']

    features, labels = create_word_level_dataset()

    print(len(features))
    print(len(labels))

    tmp_list = []

    for i in range(len(labels)):
        print(labels[i], " : ", classes_list[labels[i]])
        tmp_list.append(classes_list[labels[i]])

    # Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
    one_hot_encoded_labels = to_categorical(labels)

    print(set(tmp_list))
    print(one_hot_encoded_labels)
    #print(one_hot_encoded_labels)