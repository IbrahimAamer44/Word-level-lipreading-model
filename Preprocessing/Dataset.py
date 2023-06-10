from videos import read_and_segment_video
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

def create_word_level_dataset(speaker_id, SEQUENCE_LENGTH, CLASSES_LIST):
    sentences = get_sentences('..\\Dictionary\\roman_urdu_sentences.txt')

    # With 'sil'
    #classes_list =  ['س','ہے', 'کیسے', 'چار', 'دو', 'چھے', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نوں', 'ہاں', 'میں', 'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']
    # Without 'sil'
    #classes_list =  ['ہے', 'کیسے', 'چار', 'دو', 'چھے', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نوں', 'ہاں', 'میں', 'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']

    features = []
    labels = []
    input_len = []# List to hold each features input length(no. of frames)

    id = speaker_id

    # TEMP 3 >>
    for i in range(len(sentences)):


        #print("Speaker ID : ", id)
        path = "..\\Dataset\\Urdu\\"+str(id)+"\\" + sentences[i]
        print(i, ". ", sentences[i])

        # Extracting features from original video
        clips, lbl,inp_len = read_and_segment_video(path, SEQUENCE_LENGTH, augment=False)
        features.extend(clips)
        # Getting index of current class
        classes_index = get_classes_indexes(lbl, CLASSES_LIST)
        labels.extend(classes_index)
        input_len.extend(inp_len)

        # Extracting features from Augmented video
        clips_aug, lbl_aug, inp_len_aug = read_and_segment_video(path, SEQUENCE_LENGTH, augment=True)
        features.extend(clips_aug)
        # Getting index of current class
        classes_index_aug = get_classes_indexes(lbl_aug, CLASSES_LIST)
        labels.extend(classes_index_aug)
        input_len.extend(inp_len_aug)

        # TEMP
        #if i == 4:
        #    break


    # Converting the list to numpy arrays
    #features = np.asarray(features)
    #labels = np.array(labels)

    return features, labels


"""
    Function receives a list of labels e.g. ['<sil>', 'wo', 'kese', 'hoon', 'jee', 'kab', 'ek']
    and converts it into corresponding list containing index of each label in the list of all classes
"""
def get_classes_indexes(labels, classes_list):

    classes_indexes = []

    for lbl in labels:
        classes_indexes.append(classes_list.index(lbl))

    return classes_indexes


if __name__ == "__main__":

    classes_list =  ['ہے', 'کیسے', 'چار', 'دو', 'چھے', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نوں', 'ہاں', 'میں', 'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']

    SEQUENCE_LENGTH = 15
    speaker_id = 0

    features, labels = create_word_level_dataset(0,18,classes_list)

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