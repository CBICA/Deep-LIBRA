import warnings
warnings.filterwarnings("ignore")

# From python packages
import numpy as np
import pandas as pd
from time import time
from termcolor import colored
from collections import Counter
import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2, os, argparse, pdb, logging, pickle, json

# From my packages
from timeout import timeout
from segmentation_tools import FSLIC
from pyradiomics_features import extract_breast_radiomics_features
from breast_needed_functions import bring_back_images_to_orginal_orientation, Z_scoring
from breast_needed_functions import bring_back_images_to_orginal_size, Normalize_Image, fix_ratio




################################## This script is for training the svm
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output_path", required=False, default='./',
                help="path for saving results file")

ap.add_argument("-i", "--input", required=False, default='./file.dcm',
                help="path for input files")

ap.add_argument("-if", "--image_format", required=False, default='.png',
                help="The image format for saving")

ap.add_argument("-po", "--print_off", type=int, default=0,
                help="If this is one, it turns off printing")

ap.add_argument("-ar", "--A_Range", type=int, default=2**8-1,
                help="The number of bits for saving image")

ap.add_argument("-fis", "--final_image_size", type=int, default=512,
                help="The final size of image")

ap.add_argument("-sfn", "--saving_folder_name", default="breast_density",
                help="The name of folder that the resutls to be saved for batch processing")

ap.add_argument("-cn", "--case_name", default="Case_ID",
                help="This name defines the saving path")

ap.add_argument("-lt", "--libra_training", default="0",
                help="Zero means to masking and one means training")

ap.add_argument("-pttm", "--Path_to_trained_model", default="/cbica/home/hajimago/Net/density/model.pkl",
                help="This is path to trained model for density prediction")

ap.add_argument("-rii", "--remove_intermediate_images",
                default="K", help="R is removing and K is keeping them")

ap.add_argument("-to", "--timeout_sec", type=int, default=1800,
                help="timeout for each batch")

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
        self.libra_training = args["libra_training"]

        self.A_Range = args["A_Range"]
        self.final_image_size = args["final_image_size"]
        self.print_off = int(args["print_off"])

        self.remove_intermediate_images = args["remove_intermediate_images"]

        if self.A_Range==2**16-1:
            self.bits_conversion = "uint16"
        elif self.A_Range==2**32-1:
            self.bits_conversion = "uint32"
        else:
            self.bits_conversion = "uint8"


        self.Path_to_trained_model = args["Path_to_trained_model"]
        self.path_to_svm, _ = os.path.split(self.Path_to_trained_model)
        self.Path_to_final_feature_list = os.path.join(self.path_to_svm, "Final_Features_List.txt")
        self.Path_to_max_min = os.path.join(self.path_to_svm, "Max_Min.csv")

        self.multi_svm = True



    @timeout(args["timeout_sec"])
    def Main_Loop_Function(self):
        ######################################################################## Couple of
        ######################################################################## initializations
        T_Start = time()

        log_path = os.path.join(self.output_path, self.case_name, "LIBRA_"+self.case_name+".log")
        logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', filename=log_path, level=logging.INFO)
        logging.info('All segmentation tasks are done.')
        logging.info('Trying to calculate breast density.')


        #################################################################### Loading Image
        #################################################################### & files
        self.image = cv2.imread(self.Case_path, 0)
        self.mask = self.image>self.image.min()


        org_image_path = os.path.join(self.output_path, self.case_name,
            "air_breast_mask", self.case_name+"_16bits_Orginal"+self.image_format)
        org_image = cv2.imread(org_image_path, -1)
        org_image = fix_ratio(org_image,
                            self.final_image_size, self.final_image_size)

        air_mask_path = os.path.join(self.output_path, self.case_name,
            "air_breast_mask", self.case_name+"_air_breast_mask"+self.image_format)
        air_mask = cv2.imread(air_mask_path, -1)


        org_image = Normalize_Image(org_image, 2**13-1,
                        bits_conversion="uint16",
                        flag_min_edition=True,
                        flag_max_edition=True,
                        Min=org_image.min(),
                        Max=org_image[air_mask>0].max())


        self.image_temp = org_image.copy()
        self.image_temp[np.logical_not(self.mask)] = self.image_temp[self.mask].min()
        self.image_temp = Normalize_Image(self.image_temp, 2**8-1, bits_conversion="uint8")
        self.image_slic = np.concatenate((self.image_temp.reshape([self.final_image_size, self.final_image_size, 1]),
                                     self.image_temp.reshape([self.final_image_size, self.final_image_size, 1])), axis=2)
        self.image_slic = np.concatenate((self.image_slic,
                                     self.image_temp.reshape([self.final_image_size, self.final_image_size, 1])), axis=2)


        Header_csv_path = os.path.join(self.output_path, self.case_name, "Headers.csv")
        Header_csv = pd.read_csv(Header_csv_path, sep=',', index_col=0)


        NumSLIC = 512; ComSLIC = 1.5; SigSLIC = 0.1
        if (Header_csv["PresentationIntentType"]=="FOR PRESENTATION").any():
            ComSLIC = 5

        self.segments, Fusied_Image = FSLIC(self.image_slic, self.image_slic,
                            NumSLIC=NumSLIC, ComSLIC=ComSLIC, SigSLIC=SigSLIC, Initial=True)

        super_pixels = os.path.join(self.output_path, "superpixels")
        if not(os.path.isdir(super_pixels)): os.makedirs(super_pixels)
        super_pixels = os.path.join(super_pixels, self.case_name+self.image_format)
        cv2.imwrite(super_pixels, (Fusied_Image*255).astype("uint8"))



        self, self.segments = extract_breast_radiomics_features(self, org_image, self.mask, self.segments, self.case_name)



        Saving_Path_All = os.path.join(self.output_path, self.saving_folder_name)
        if not(os.path.isdir(Saving_Path_All)): os.makedirs(Saving_Path_All)

        Unq_columns_listed = (np.unique(self.FEATUREs.columns)).tolist()
        columns_listed = self.FEATUREs.columns.tolist()
        Repeated_Columns = (Counter(columns_listed) - Counter(set(columns_listed))).keys()


        #fix repeated columns names
        for Column in Repeated_Columns:
            Indexes = np.argwhere(self.FEATUREs.columns==Column)[1:]
            for N, Index in enumerate(Indexes):
                self.FEATUREs.columns.values[Index] = self.FEATUREs.columns[Index]+"."+str(N+1)


        if self.libra_training != "1":
            ##### do density mapping
            Pixel_Spacing = Header_csv["ImagerPixelSpacing"] # remember this format "['0.094090909', '0.094090909']"
            Coma_loc = Pixel_Spacing[0].find(',')
            Pixel_Spacing_X = float(Pixel_Spacing[0][2:Coma_loc-1])
            Pixel_Spacing_Y = float(Pixel_Spacing[0][Coma_loc+3:-2])
            pixel_to_cm_conversion = Pixel_Spacing_X * Pixel_Spacing_Y * 0.1 * 0.1


            if self.multi_svm:
                for SVM_INDEX in range(3):
                    SVM_INDEX += 1
                    Base, File = os.path.split(self.Path_to_trained_model)
                    with open(os.path.join(Base, str(SVM_INDEX)+File), 'rb') as pickle_file:
                        loaded_model = pickle.load(pickle_file)
                    Base, File = os.path.split(self.Path_to_final_feature_list)
                    with open(os.path.join(Base, str(SVM_INDEX)+File), 'r') as json_file:
                        feature_list = json.load(json_file)
                    Base, File = os.path.split(self.Path_to_max_min)
                    max_min = pd.read_csv(os.path.join(Base, str(SVM_INDEX)+File), sep=',', index_col=0)
                    max_min = max_min.loc[feature_list]

                    self.normalized_features_svm = (self.FEATUREs[feature_list]-max_min["Min"])/(max_min["Max"]-max_min["Min"])
                    temp_segment_Classes = loaded_model.predict(self.normalized_features_svm)

                    if SVM_INDEX==1:
                        segment_Classes = temp_segment_Classes.copy()
                    else:
                        segment_Classes += temp_segment_Classes
                segment_Classes = segment_Classes/3.0
                segment_Classes = np.int16(np.round(segment_Classes))

            else:
                with open(self.Path_to_trained_model, 'rb') as pickle_file:
                    loaded_model = pickle.load(pickle_file)
                with open(self.Path_to_final_feature_list, 'r') as json_file:
                    feature_list = json.load(json_file)
                max_min = pd.read_csv(self.Path_to_max_min, sep=',', index_col=0)
                max_min = max_min.loc[feature_list]

                self.normalized_features_svm = (self.FEATUREs[feature_list]-max_min["Min"])/(max_min["Max"]-max_min["Min"])
                segment_Classes = loaded_model.predict(self.normalized_features_svm)


            self.FEATUREs["Segment_Class"] = segment_Classes

            ## This is where I might need to modify the density if it is really low
            # np.argwhere(segment_Classes==1)

            breast_area = self.FEATUREs["Breast_area"].iloc[0]

            BD = np.sum(self.FEATUREs["Seg_area"][segment_Classes>0]/breast_area)

            self.FEATUREs["Breast_Density_Percentage"] = BD

            print(self.case_name, BD)

            # save image desnity map
            self.mask_density = np.zeros(self.image.shape)
            Indexes = self.FEATUREs["Seg_index"]
            for  Index in Indexes[self.FEATUREs["Segment_Class"]==1]:
                self.mask_density[self.segments==Index] = 255


            Path_to_csv_ori = os.path.join(self.output_path, self.case_name, "Headers.csv")
            Path_to_csv_size = os.path.join(self.output_path, self.case_name, "air_breast_mask", "fixing_ratio.csv")


            self.mask = self.mask*255
            contours_mask, _ = cv2.findContours(self.mask.astype("uint8"),
                                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            fig = plt.figure(frameon=False)
            fig.set_size_inches(5, 5)
            ax = plt.gca()
            for contour in contours_mask:
                if len(contour)>1:
                    ax.imshow(self.image, 'gray')
                    contour = np.concatenate((contour[:,:,0].T, contour[:,:,1].T), axis=0)
                    ax.plot(contour[0], contour[1], linewidth=3, color='r')

            contours_density, _ = cv2.findContours(self.mask_density.astype("uint8"),
                                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_density:
                contour = np.concatenate((contour[:,:,0].T, contour[:,:,1].T), axis=0)
                ax.plot(contour[0], contour[1], linewidth=2, color='lime')
            ax.set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            ax.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            image_path = os.path.join(self.output_path, self.case_name, self.case_name+
                                      "_dense_tissue_overlay_on_image"+self.image_format)
            fig.canvas.draw()
            plt.close()

            image_returned = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_returned = image_returned.reshape(fig.canvas.get_width_height()[::-1] + (3,)).astype("uint8")
            image_returned = cv2.cvtColor(image_returned, cv2.COLOR_BGR2RGB)
            image_returned= bring_back_images_to_orginal_size(Path_to_csv_size, image_returned, "image")
            image_returned = bring_back_images_to_orginal_orientation(Path_to_csv_ori, image_returned)

            Path_to_org_image = os.path.join(self.output_path, self.case_name, "air_breast_mask",
                                             self.case_name+"_Normalized"+self.image_format)
            original_image = cv2.imread(Path_to_org_image, 0)
            original_image = cv2.cvtColor(original_image,cv2.COLOR_GRAY2RGB)
            original_image = bring_back_images_to_orginal_size(Path_to_csv_size, original_image, "image")
            original_image = bring_back_images_to_orginal_orientation(Path_to_csv_ori, original_image)

            original_image[image_returned==255] = self.A_Range
            original_image[image_returned>0] = image_returned[image_returned>0]



            Ch1 = original_image[...,0]
            Ch2 = original_image[...,1]
            Ch3 = original_image[...,2]
            Ch3[image_returned[...,1]==255] = 0
            Ch2[image_returned[...,2]==255] = 0
            Ch1[np.logical_or(image_returned[...,2]==255, image_returned[...,1]==255)] = 0

            Ch1[np.logical_and(image_returned[...,0]==255, image_returned[...,2]==255, image_returned[...,1]==255)] = 255
            Ch2[np.logical_and(image_returned[...,0]==255, image_returned[...,2]==255, image_returned[...,1]==255)] = 255
            Ch3[np.logical_and(image_returned[...,0]==255, image_returned[...,2]==255, image_returned[...,1]==255)] = 255

            original_image[...,0] = Ch1
            original_image[...,1] = Ch2
            original_image[...,2] = Ch3
            cv2.imwrite(image_path, original_image)

            final_mask = os.path.join(Saving_Path_All, self.case_name+self.image_format)
            cv2.imwrite(final_mask, original_image)


            dense_mask_file_name = os.path.join(self.output_path, self.case_name, self.case_name+
                                                "_dense_tissue_mask"+self.image_format)
            self.mask_density = bring_back_images_to_orginal_size(Path_to_csv_size, self.mask_density)
            self.mask_density = bring_back_images_to_orginal_orientation(Path_to_csv_ori, self.mask_density)
            cv2.imwrite(dense_mask_file_name, self.mask_density)


            final_mask = os.path.join(self.output_path, self.case_name, self.case_name+
                                      "_final_breask_mask_image_size"+self.image_format)
            self.mask = bring_back_images_to_orginal_size(Path_to_csv_size, self.mask)
            self.mask = bring_back_images_to_orginal_orientation(Path_to_csv_ori, self.mask)
            cv2.imwrite(final_mask, self.mask)


            self.FEATUREs["Breast_area"] *= pixel_to_cm_conversion
            self.FEATUREs["Seg_area"] *= pixel_to_cm_conversion


            features_file_name = os.path.join(Saving_Path_All, self.case_name+"_Features.csv")
            self.FEATUREs_new = self.FEATUREs[self.FEATUREs.columns[:102]].iloc[[0]]
            self.FEATUREs_new[self.FEATUREs.columns[-1]] = self.FEATUREs[self.FEATUREs.columns[-1]].iloc[0]
            self.FEATUREs_new["Dense_area"] = np.sum(self.FEATUREs["Seg_area"][self.FEATUREs["Segment_Class"]==1])
            self.FEATUREs_new.index = [self.case_name]
            self.FEATUREs_new.to_csv(features_file_name)

        else:
            features_file_name = os.path.join(Saving_Path_All, self.case_name+"_Features.csv")
            self.FEATUREs.to_csv(features_file_name)


        features_file_name = os.path.join(self.output_path, self.case_name, self.case_name+"_Features.csv")
        self.FEATUREs.to_csv(features_file_name)



        if self.remove_intermediate_images=="R":
            try:
                os.remove(org_image_path)
            except:
                org_image_path



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
