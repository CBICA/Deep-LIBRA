# This is read and write functions needed for deeplearning
import numpy as np
from glob import glob
from copy import deepcopy
import os, pdb, json, cv2
from shutil import copyfile
import matplotlib.pyplot as plt
from keras.callbacks import BaseLogger
from keras.preprocessing.image import ImageDataGenerator



def get_image_info(obj):
    # first image info
    if hasattr(obj, "image_folder"):
        Image_Path = os.path.join(obj.train_path, obj.image_folder)
    else:
        Image_Path = obj.train_path

    Image_Path = os.path.join(Image_Path, "*"+obj.image_format)

    Images_files = sorted(glob(Image_Path))

    if len(Images_files)>=1:
        Image_sample = cv2.imread(Images_files[0], -1)
    else:
        Image_sample = cv2.imread(Images_files, -1)

    obj.target_size = (Image_sample.shape[0], Image_sample.shape[1])
    if len(Image_sample.shape)>2:
        obj.image_dimension = Image_sample.shape[2]
        obj.image_color_mode = "rgb"
    else:
        obj.image_dimension = 1
        obj.image_color_mode = "grayscale"


    obj.num_training_image = len(Images_files)

    # then mask info
    if hasattr(obj, "mask_folder"):
        Mask_Path = os.path.join(obj.train_path, obj.mask_folder)
        Mask_Path = os.path.join(Mask_Path, "*"+obj.image_format)

        Mask_Files = sorted(glob(Mask_Path))
        Mask_sample = cv2.imread(Mask_Files[0])

        if len(Image_sample.shape)>2:
            obj.mask_color_mode = "rgb"
        else:
            obj.mask_color_mode = "grayscale"

    return(obj)



def adjustData(image, obj, mask=[]):
    # make image between zero to one
    # plt.imshow(image[0,...,0]);plt.show()

    if image.max()>1:
        image = image/image.max()

    if len(image.shape)!=4:
        temp = deepcopy(image)
        image = np.zeros(image.shape + (1, ))
        image[..., 0] = temp


    if len(mask)!=0:
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]

        Steps_fidning_classes = 256/obj.num_class
        Bottom_range_class = 0

        # add a dimension for mask
        if len(mask.shape) == 3:
            new_mask = np.zeros(mask.shape + (obj.num_class, ))

        for i in range(obj.num_class):
            new_mask[np.logical_and(mask>=Bottom_range_class,
                        mask<Bottom_range_class+Steps_fidning_classes), i] = 1
            Bottom_range_class += Steps_fidning_classes

        mask = new_mask

    return (image, mask)



def trainGenerator(obj, val_path=[]):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**obj.aug_dict)
    mask_datagen = ImageDataGenerator(**obj.aug_dict)

    if len(val_path) == 0:
        General_path_Image = obj.train_path
    else:
        General_path_Image = val_path

    image_generator = image_datagen.flow_from_directory(
        General_path_Image,
        classes = [obj.image_folder],
        class_mode = None,
        color_mode = obj.image_color_mode,
        target_size = obj.target_size,
        batch_size = obj.batch_size,
        seed = obj.seed)
    #     save_to_dir = "/home/ohm/AUGMENT",
    #     save_prefix='aug',
    #     save_format='png'

    mask_generator = mask_datagen.flow_from_directory(
        General_path_Image,
        classes = [obj.mask_folder],
        class_mode = None,
        color_mode = obj.mask_color_mode,
        target_size = obj.target_size,
        batch_size = obj.batch_size,
        seed = obj.seed)

    train_generator = zip(image_generator, mask_generator)
    for (image, mask) in train_generator:
        image, mask = adjustData(image, obj, mask)
        # cv2.imwrite("/home/ohm/image.png", np.uint8(image[0,...,0]*255))
        # cv2.imwrite("/home/ohm/mask.png", np.uint8(mask[0,...,1]*255))
        yield (image, mask)



def train_generator_classify(obj, labels, val_path=[]):
    image_datagen = ImageDataGenerator(**obj.aug_dict)

    if len(val_path) == 0:
        General_path_Image = obj.train_path
    else:
        General_path_Image = val_path

    image_generator = image_datagen.flow_from_directory(
        General_path_Image,
        class_mode = "categorical",
        color_mode = obj.image_color_mode,
        target_size = (int(obj.image_final_size), int(obj.image_final_size)),
        batch_size = obj.batch_size)

    return(image_generator)



def prepare_kfold(obj, images_path, fold_path, mode, indexes):
    fold_masks_path = os.path.join(fold_path, mode, obj.mask_folder)
    fold_images_path = os.path.join(fold_path, mode, obj.image_folder)
    fold_org_images_path = os.path.join(fold_path, mode, obj.org_image_folder)


    if not(os.path.isdir(fold_masks_path)): os.makedirs(fold_masks_path)
    if not(os.path.isdir(fold_images_path)): os.makedirs(fold_images_path)
    if not(os.path.isdir(fold_org_images_path)): os.makedirs(fold_org_images_path)


    for n, index in enumerate(indexes.astype(int)):
        copyfile( os.path.join( os.path.join(images_path, obj.mask_folder),
                "%05d.png" %(n)), os.path.join(fold_masks_path, "%05d.png" %(n)) )
        copyfile( os.path.join( os.path.join(images_path, obj.image_folder),
                "%05d.png" %(n)), os.path.join(fold_images_path, "%05d.png" %(n)) )

        if os.path.isfile(fold_masks_path):
            copyfile( os.path.join( os.path.join(images_path, obj.org_image_folder),
                "%05d.png" %(n)), os.path.join(fold_org_images_path, "%05d.png" %(n)) )



class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        # plot the training loss and accuracy
        colors = ['g', 'b', 'm', 'r', 'c', 'y', "lime", "orange", "gray", "navy", "brown", "pink"]
        number_color = 0
        font_size = 10
        if len(self.H["loss"]) > 1 and len(self.H["loss"])%2 == 0.0:
            N = np.arange(0, len(self.H["loss"]))
            fig = plt.figure()
            fig.set_size_inches(10, 4)
            plt.rcParams['savefig.facecolor'] = "0.6"
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            ax = fig.add_subplot(1, 1, 1)

            for key in self.H.keys():
                if key!='loss' and key!='val_loss':
                    ax.plot(N, self.H[key], label=key, color=colors[number_color])
                    number_color += 1

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(font_size)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(font_size)

            plt.title("Performance Measures [Epoch {}]".format(len(self.H["loss"])), color='k', fontweight='bold')
            plt.xlabel("Epoch #", color='k')
            plt.ylabel("Measures Values", color='k')
            plt.legend()

            plt.savefig(self.figPath, dpi=400)
            plt.close()



def testGenerator(obj):
    image_datagen = ImageDataGenerator()
    General_path_Image = obj.train_path

    Path, file_name = os.path.split(General_path_Image)
    if file_name == "image":
        General_path_Image = Path

    image_generator = image_datagen.flow_from_directory(
        General_path_Image,
        classes = ["image"], # always assuming that images are in path/image
        color_mode = obj.image_color_mode,
        class_mode = None,
        target_size = (obj.image_final_size, obj.image_final_size),
        batch_size = obj.batch_size,
        shuffle = False)

    return(image_generator)



def eval_image_Generator(obj):
    image_datagen = ImageDataGenerator()
    General_path_Image = obj.train_path

    Path, file_name = os.path.split(General_path_Image)
    if file_name == "image":
        General_path_Image = Path

    image_generator = image_datagen.flow_from_directory(
        General_path_Image,
        classes = ["image"], # always assuming that images are in image folder
        class_mode = None,
        color_mode = obj.image_color_mode,
        target_size = (obj.image_final_size, obj.image_final_size),
        batch_size = obj.batch_size,
        shuffle = False)

    for image, M in zip(image_generator, image_generator):
        image = image / image.max()

        yield image


def eval_mask_Generator(obj):
    mask_datagen = ImageDataGenerator()
    General_path_Image = obj.train_path

    mask_generator = mask_datagen.flow_from_directory(
        General_path_Image,
        classes = [obj.mask_folder],
        class_mode = None,
        color_mode = obj.image_color_mode,
        target_size = (obj.image_final_size, obj.image_final_size),
        batch_size = obj.batch_size,
        shuffle = False)

    for mask, M in zip(mask_generator, mask_generator):
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask = np.round( mask/mask.max()*(obj.num_class-1) )

        # add a dimension for mask
        new_mask = np.zeros(mask.shape + (obj.num_class, ))
        for i in range(obj.num_class):
            new_mask[mask==i, i] = 1
        mask = new_mask

        yield mask



def saveResult(obj, saving_name="_predict"):
    if hasattr(obj, "image_folder"):
        Image_Path = os.path.join(obj.train_path, obj.image_folder)
    else:
        Image_Path = obj.train_path

    Images_paths = sorted(glob(os.path.join(Image_Path, "*"+obj.image_format)))
    for i, (image, input_image_path) in enumerate( zip(obj.results, Images_paths) ):
        if obj.num_class>3:
            image[:, :, 0] = 0
            for Class in range(image.shape[-1]-1):
                image[:, :, 0] = image[:, :, 0]+image[:, :, Class+1]*(Class+1)
            image = image[:, :, 0]
        elif obj.num_class==2:
            image = image[:, :, 1]

        image = np.round(image)
        image = np.uint8(image/image.max()*obj.A_Range)
        _, file_name = os.path.split(input_image_path)
        cv2.imwrite(os.path.join(obj.saving_path, file_name[:-4]+saving_name+obj.image_format), image)


def saveResults_batch_based(obj, results, image_names, saving_name="_predict"):
    for image, input_image_path in zip(results, image_names):
        if obj.num_class>3:
            for Class in range(image.shape[-1]-1):
                image[:, :, 0] = image[:, :, 0]+image[:, :, Class+1]*(Class+1)
            image = image[:, :, 0]
        elif obj.num_class==2:
            image = image[:, :, -1]

        image = np.uint8(image/image.max()*obj.A_Range)
        image = (image>200).astype("uint8")

        _, file_name = os.path.split(input_image_path)
        file_name = file_name[:-4]

        cv2.imwrite(os.path.join(obj.saving_path, file_name+saving_name+obj.image_format), image)
        # cv2.imwrite(os.path.join(obj.output_path, file_name, file_name+saving_name+obj.image_format), image)
