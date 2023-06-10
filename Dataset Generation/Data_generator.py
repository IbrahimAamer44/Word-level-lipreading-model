from Dataset import create_word_level_dataset
if __name__ == "__main__":
    classes_list = ['ہے', 'کیسے', 'چار', 'دو', 'چھے', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نوں', 'ہاں', 'میں',
                    'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']

    SEQUENCE_LENGTH = 15
    speaker_id = 0

    features, labels = create_word_level_dataset(0, 18, classes_list)

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
    # print(one_hot_encoded_labels)