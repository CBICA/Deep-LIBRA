# From python packages
from time import time
from termcolor import colored
import cv2, os, argparse, pydicom, logging


# From my packages
from breast_needed_functions import Normalize_Image
from breast_needed_functions import object_oriented_preprocessing, Remove_Top_Below_Side_effect

################################## This script is for training the svm
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output_path", required=False, default='./output',
                help="path for saving results file")

ap.add_argument("-i", "--input_dicom", required=False, default='Full_path_to_dicom_file',
                help="path for input files")

ap.add_argument("-if", "--image_format", required=False, default='.png',
                help="The image format for saving")


args = vars(ap.parse_args())



class Segmentor(object): # The main class
    def __init__(self):
        ######################################################################## Initial
        ######################################################################## Values
        self.Case = args["input_dicom"]
        self.output_path = args["output_path"]
        self.image_format = args["image_format"]



    def Main_Loop_Function(self):
        ######################################################################## Couple of
        ######################################################################## initializations
        T_Start = time()
        # no output path = return the results in the same path as dataset
        if self.output_path == '0':
            self.output_path = self.PATH

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


        #################################################################### Loading Image
        #################################################################### & files
        # Read Dicom file
        try:
            self.ds = pydicom.dcmread(self.Case)
            self.image = (self.ds.pixel_array).astype("float")
        except:
            ############ FIX THIS
            from medpy.io import load
            self.image, self.ds = load(self.Case)


        # Preprocessing step
        self = object_oriented_preprocessing(self)
        self = Remove_Top_Below_Side_effect(self)


        #################################################################### making
        #################################################################### the mask and original image
        self.image_16bits = Normalize_Image(self.image, 2**16-1,
                        bits_conversion="uint16", flag_min_edition=True, Min=self.image.min())


        Save_name_img = os.path.join(Image_Path, "air_breast_mask",
                                     File+"_16bits_Orginal"+self.image_format)
        cv2.imwrite(Save_name_img, self.image_16bits)




###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    Info = Segmentor()
    Info.Main_Loop_Function()
