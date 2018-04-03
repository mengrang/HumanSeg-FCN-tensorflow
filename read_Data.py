import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
from datetime import date
import TensorflowUtils as utils
def read_dataset(data_dir):
    pickle_filename = "humanseg.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        pickle_filename = "humanseg.pickle"
        pickle_filepath = os.path.join(data_dir, pickle_filename)
        result = create_image_lists(data_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f)
    else:
        print ("Found pickle file!")
    open(pickle_filepath, 'a').close()
    scores = {}
    try:
        with open(pickle_filepath, "rb") as file:
            unpickler = pickle.Unpickler(file)
            scores = unpickler.load()
            if not isinstance(scores, dict):
                scores = {}
    except EOFError:
        return {}
    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['val']
        del result
    # print(training_records[:10])
    return training_records, validation_records


def create_image_lists(data_dir):
    if not gfile.Exists(data_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    training=[]
    val=[]
    image_file= []
    annot_file = []
    image_list = {'training': training, 'val': val}
    data_split_dir=data_dir
    image_dir = os.path.join(data_split_dir, 'images')
    annot_dir = os.path.join(data_split_dir, 'annotations')
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            filename = os.path.splitext(file)[0]
            image_per_id=file
            annot_per_id =filename+'-mask.jpg'
            image_file.append(image_per_id)
            annot_file.append(annot_per_id)
    image_file = sorted(image_file)
    annot_file = sorted(annot_file)
    val_image_file = image_file[40000:]
    for i in val_image_file:
        f = os.path.join(image_dir, i)
        val_filename = os.path.splitext(i)[0]
        val_annotation_file = os.path.join(annot_dir, val_filename + '-mask.jpg')
        val_record = {'image': f, 'annotation': val_annotation_file, 'filename': val_filename}
        val.append(val_record)
    random.shuffle(image_list['val'])

    tra_image_file=image_file[:39999]
    for i in tra_image_file:
        f=os.path.join(image_dir,i)
        tra_filename = os.path.splitext(i)[0]
        tra_annotation_file= os.path.join(annot_dir, tra_filename + '-mask.jpg')
        tra_record = {'image':f, 'annotation': tra_annotation_file, 'filename': tra_filename}
        training.append(tra_record)
    random.shuffle(image_list['training'])
    print(image_list['training'][:10])
    return image_list
if __name__=='__main__':
    data_dir='/home/hongyvsvxinlang/git/human_seg/data'
    create_image_lists(data_dir)
    read_dataset(data_dir)
