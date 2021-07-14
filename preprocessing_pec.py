import warnings
warnings.filterwarnings("ignore")

# From python packages
import numpy as np
from time import time
from skimage import exposure
from termcolor import colored
import cv2, os, argparse, pdb, logging


# From my packages
from breast_needed_functions import find_logical_background_objs
from breast_needed_functions import Normalize_Image, find_largest_obj


################################## This script is for training the svm
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output_path", required=False, default='./output',
                help="path for saving results file")

ap.add_argument("-i", "--input", required=False, default='Full_path_to_image_name',
                help="path for input files")

ap.add_argument("-if", "--image_format", required=False, default='.png',
                help="The image format for saving")

ap.add_argument("-po", "--print_off", type=int, default=0,
                help="If this is one, it turns off printing")

ap.add_argument("-ar", "--A_Range", type=int, default=2**8-1,
                help="The number of bits for saving image")

ap.add_argument("-fis", "--final_image_size", type=int, default=512,
                help="The final size of image")

ap.add_argument("-sfn", "--saving_folder_name", default="pec_net_data/image",
                help="The name of folder that the resutls to be saved for batch processing")

ap.add_argument("-cn", "--case_name", default="Case_ID",
                help="This name defines the saving path")

args = vars(ap.parse_args())



class Segmentor(object): # The main class
    def __init__(self):
        ######################################################################## Initial
        ######################################################################## Values
        self.Case_path = args["input"]
        self.image_format = args["image_format"]
        self.saving_folder_name = args["saving_folder_name"]
        self.case_name = args["case_name"]
        self.output_path = args["output_path"]

        self.A_Range = args["A_Range"]
        self.final_image_size = args["final_image_size"]
        self.print_off = int(args["print_off"])

        if self.A_Range==2**16-1:
            self.bits_conversion = "uint16"
        elif self.A_Range==2**32-1:
            self.bits_conversion = "uint32"
        else:
            self.bits_conversion = "uint8"

        self.threshold_percentile = 0.5



    def Main_Loop_Function(self):
        ######################################################################## Couple of
        ######################################################################## initializations
        T_Start = time()

        log_path = os.path.join(self.output_path, self.case_name, "LIBRA_"+self.case_name+".log")
        logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', filename=log_path, level=logging.INFO)
        logging.info('Segmentation of background from breast+pectrocal is done.')
        logging.info('Segmentation of breast from pectrocal starting.')


        #################################################################### Loading Image
        #################################################################### & files
        org_image_path = os.path.join(self.output_path, self.case_name,
                "air_breast_mask", self.case_name+"_Normalized"+self.image_format)
        self.org_image = cv2.imread(org_image_path, -1)


        self.mask = cv2.imread(self.Case_path, 0)
        self.mask = self.mask>0


        try:
            # join masks
            self.mask = find_logical_background_objs(self.mask)
            self.mask = find_largest_obj(self.mask)

        except:
            if self.print_off==0:
                print("1 THIS IMAGE HAD ISSUE FOR PEC PREPROCESSING: "+self.case_name)
            logging.info("The breast air CNN mask had an issue. Warning.")

            self.mask = find_largest_obj(self.mask)


        self.org_image[np.logical_not(self.mask)] = 0
        try:
            Min = self.org_image[self.mask].min()
        except:
            Min = self.org_image.min()
        self.org_image[np.logical_not(self.mask)] = Min


        non_zero = self.org_image[self.org_image>Min]
        if len(non_zero)>0:
            self.image = (self.org_image-(np.percentile(non_zero,self.threshold_percentile)-1/self.A_Range))/ (
                non_zero.max()-(np.percentile(non_zero,self.threshold_percentile)-1/self.A_Range))
        else:
            self.image = self.org_image/self.org_image.max()


        self.image[self.image<0] = 0
        self.image = self.image * self.A_Range
        self.image = self.image.astype(self.bits_conversion)


        self.image_he = exposure.equalize_hist(self.image, nbins=self.A_Range,
                                               mask=self.mask>0)
        self.image_he[self.mask==0]=0
        self.image_he = Normalize_Image(self.image_he, self.A_Range-1,
                                          bits_conversion=self.bits_conversion)
        self.image_he += 1
        self.image_he[self.mask==0]=0


        self.org_image[self.mask==0] = 0
        self.image_main = Normalize_Image(self.org_image, self.A_Range-1,
                                          bits_conversion=self.bits_conversion)
        self.image_main += 1
        self.image_main[self.mask==0] = 0

        self.image = np.concatenate((self.image.reshape([self.final_image_size, self.final_image_size, 1]),
                                     self.image_he.reshape([self.final_image_size,
                                     self.final_image_size,1])), axis=2)
        self.image = np.concatenate((self.image,
                                     self.image_main.reshape([self.final_image_size,
                                     self.final_image_size,1])), axis=2)
        self.image = self.image.astype(self.bits_conversion)


        Image_Path = os.path.join(self.output_path, self.case_name)
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        Mask_path = os.path.join(Image_Path, "air_breast_mask")
        if not(os.path.isdir(Mask_path)): os.makedirs(Mask_path)
        Save_name_mask = os.path.join(Mask_path, self.case_name+"_air_breast_mask"+self.image_format)
        cv2.imwrite(Save_name_mask, self.mask.astype(self.bits_conversion)*self.A_Range)


        Image_Path = os.path.join(Image_Path, "breast_mask")
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        Save_name_img = os.path.join(Image_Path, self.case_name+"_pec_breast_preprocessed"+self.image_format)
        cv2.imwrite(Save_name_img, self.image)


        Image_Path = os.path.join(self.output_path, self.saving_folder_name)
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        Save_name_img = os.path.join(Image_Path, self.case_name+self.image_format)
        cv2.imwrite(Save_name_img, self.image)


        logging.info("The path of saved image is: "+Save_name_img)
        if self.print_off==0: print("[INFO] The path of saved image is: "+Save_name_img)


        T_End = time()
        if self.print_off==0: print("[INFO] Elapsed Time (for this file): "+'\033[1m'+ \
              colored(str(round(T_End-T_Start, 2)), 'blue')+'\033[0m'+" seconds")

        logging.info("The breast-air segmentation was successfully processed for this case.")
        if self.print_off==0: print(colored("[INFO]", 'green')+" The breast-air segmentation was successfully processed for this case.")



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    Info = Segmentor()
    Info.Main_Loop_Function()
