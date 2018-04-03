import numpy as np
import scipy.misc as misc #for pre-processing the images
import read_Data as scene_parsing

class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _get_labels(self,filename):
        image = self._transform(filename)
        img2 = image/255.
        return img2.astype(np.int32)

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        #Expand the shape of an array.Insert a new axis
        self.annotations = np.array([self._get_labels(filename['annotations']) for filename in self.files])
        # self.annotations = np.array([self._transform(filename['annotation']) for filename in self.files])
        print(self.images.shape)
        print(self.annotations.shape)

    def _transform(self, filename):#resize the images
        image = misc.imread(filename)
        # image=img_as_float(image)

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])#array([0,1,.,images.shape[0]-1])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()#to list
        return self.images[indexes], self.annotations[indexes]

if __name__=='__main__':
    data_dir = '/home/hongyvsvxinlang/git/human_seg/data'
    image_options = {'resize': True, 'resize_size': 256}
    train_records, valid_records = scene_parsing.read_dataset(data_dir)
    train_images, train_annotations = BatchDatset(train_records, image_options).next_batch(256)
