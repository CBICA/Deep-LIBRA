import warnings
warnings.filterwarnings("ignore")

# From python packages
import numpy as np
from time import time
from termcolor import colored
import cv2, os, argparse, pdb, logging

# From my packages
from breast_needed_functions import Normalize_Image, detect_buttom_portion
from breast_needed_functions import find_logical_pec_objs, find_largest_obj, fix_ratio
from breast_needed_functions import bring_back_images_to_orginal_size, bring_back_images_to_orginal_orientation

import matplotlib.pyplot as plt

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

ap.add_argument("-sfn", "--saving_folder_name", default="final_images/image",
                help="The name of folder that the resutls to be saved for batch processing")

ap.add_argument("-cn", "--case_name", default="Case_ID",
                help="This name defines the saving path")

ap.add_argument("-fb", "--find_bottom", default="1",
                help="if this is one, it tries to remove the bottom.")

ap.add_argument("-rii", "--remove_intermediate_images",
                default="K", help="R is removing and K is keeping them")

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
        self.find_bottom = args["find_bottom"]

        if self.A_Range==2**16-1:
            self.bits_conversion = "uint16"
        elif self.A_Range==2**32-1:
            self.bits_conversion = "uint32"
        else:
            self.bits_conversion = "uint8"



    def Main_Loop_Function(self):
        ######################################################################## Couple of
        ######################################################################## initializations
        T_Start = time()

        log_path = os.path.join(self.output_path, self.case_name, "LIBRA_"+self.case_name+".log")
        logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', filename=log_path, level=logging.INFO)
        logging.info('Segmentation of pectoral from breast is done.')
        logging.info('Masking final masks and normalized image.')


        #################################################################### Loading Image
        #################################################################### & files
        try:
            org_image_path = os.path.join(self.output_path, self.case_name,
                "air_breast_mask", self.case_name+"_16bits_Orginal"+self.image_format)
            self.org_image = cv2.imread(org_image_path, -1)
            self.org_image = fix_ratio(self.org_image,
                            self.final_image_size, self.final_image_size)

        except:
            org_image_path = os.path.join(self.output_path, self.case_name,
                "air_breast_mask", self.case_name+"_Normalized"+self.image_format)
            self.org_image = cv2.imread(org_image_path, -1)

        air_mask_path = os.path.join(self.output_path, self.case_name,
                "air_breast_mask", self.case_name+"_air_breast_mask"+self.image_format)
        self.mask = cv2.imread(air_mask_path, -1)

        self.pec_mask = cv2.imread(self.Case_path, -1)
        if len(self.pec_mask.shape)>2:
            self.pec_mask = self.pec_mask[...,-1]

        self.pec_mask = find_logical_pec_objs(self.pec_mask>0)
        self.mask[self.pec_mask>0] = 0
        self.mask = self.mask>0

        self.mask[:5, :] = False
        self.mask[-1, :] = False
        self.mask[:, 0] = False
        self.mask = find_largest_obj(self.mask)

        # im_floodfill = self.mask.copy()
        # im_floodfill[:3,:] = False # to make it safe
        # im_floodfill[-3:,:] = False
        # im_floodfill[:,:3] = False
        # im_floodfill[:,-3:] = False
        # loc = np.where(im_floodfill)
        # h, w = im_floodfill.shape[:2]
        # mask = np.zeros((h+2, w+2), np.uint8)
        # im_floodfill = cv2.floodFill((im_floodfill*255).astype("uint8"),
        #                              mask, (loc[0][0], loc[1][0]), 255)[1]
        # im_floodfill = cv2.bitwise_not(im_floodfill)
        # im_floodfill = im_floodfill>0
        # if np.array_equal(im_floodfill, im_floodfill.astype(bool)) and im_floodfill.any():
        #     self.mask = self.mask | np.logical_not(im_floodfill)

        if self.find_bottom == "1":
            try:
                self.mask = detect_buttom_portion(self, self.mask)
            except:
                self.mask = self.mask
            self.mask = find_largest_obj(self.mask)



        self.org_image[np.logical_not(self.mask)] = 0
        Min = self.org_image[self.mask].min()
        self.org_image[np.logical_not(self.mask)] = Min


        # replace small and too bright spots
        top_one = np.percentile(self.org_image, 99.9)
        if (self.org_image>top_one).any():
            self.org_image[self.org_image>top_one] = int(top_one)


        self.org_image = Normalize_Image(self.org_image, self.A_Range-1,
                            bits_conversion=self.bits_conversion, Name=self.case_name)+1
        self.org_image[np.logical_not(self.mask)] = 0
        self.image_main = self.image_he = self.image = self.org_image


        self.image = np.concatenate((self.image.reshape([self.final_image_size, self.final_image_size, 1]),
                                     self.image_he.reshape([self.final_image_size,
                                     self.final_image_size,1])), axis=2)
        self.image = np.concatenate((self.image,
                                     self.image_main.reshape([self.final_image_size,
                                     self.final_image_size,1])), axis=2)
        self.image = self.image.astype(self.bits_conversion)


        Image_Path = os.path.join(self.output_path, self.saving_folder_name)
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        Save_name_img = os.path.join(Image_Path, self.case_name+self.image_format)
        cv2.imwrite(Save_name_img, self.image)


        Image_Path = os.path.join(self.output_path, self.case_name)
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        Save_name_img = os.path.join(Image_Path, self.case_name+"_final_breast_notmalized_image"+self.image_format)

        Path_to_csv_size = os.path.join(self.output_path, self.case_name, "air_breast_mask", "fixing_ratio.csv")
        self.image = bring_back_images_to_orginal_size(Path_to_csv_size, self.image, type="image")
        Path_to_csv_ori = os.path.join(self.output_path, self.case_name, "Headers.csv")
        self.image = bring_back_images_to_orginal_orientation(Path_to_csv_ori, self.image)
        cv2.imwrite(Save_name_img, self.image)


        Image_Path = os.path.join(Image_Path, "breast_mask")
        if not(os.path.isdir(Image_Path)): os.makedirs(Image_Path)
        Save_name_mask = os.path.join(Image_Path, self.case_name+"_final_breast_mask"+self.image_format)
        cv2.imwrite(Save_name_mask, (self.mask*255).astype("uint8"))


        logging.info("The path of saved image is: "+Save_name_img)
        if self.print_off==0: print("[INFO] The path of saved image is: "+Save_name_img)


        T_End = time()
        if self.print_off==0: print("[INFO] Elapsed Time (for this file): "+'\033[1m'+ \
              colored(str(round(T_End-T_Start, 2)), 'blue')+'\033[0m'+" seconds")

        logging.info("The process for this case is done.")
        if self.print_off==0: print(colored("[INFO]", 'green')+" The process for this case is done.")



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    Info = Segmentor()
    Info.Main_Loop_Function()
