import numpy as np
import os, pdb, cv2
import pandas as pd
from subprocess import call
from termcolor import colored
import matplotlib.pyplot as plt
from segmentation_tools import Normalize_Image, find_largest_obj




def run_loop_multi_cpu(obj, image_path, code_path):
    Path, File = os.path.split(obj.Case)
    if File[-4:] == ".dcm": obj.File = File[:-4]

    call(["python3", os.path.join(code_path, "preprocessing.py"), "-i",
          image_path, "-o", obj.output_path, "-if", obj.image_format,
          "-po", obj.print_off, "-sfn", obj.saving_folder_name_net_air,
          "-ar", str(obj.A_Range), "-fis", str(obj.final_image_size),
          "-lsm", obj.libra_segmentation_method, "-fpm", obj.find_pacemaker])



def run_loop_multi_cpu_just_org_image(obj, image_path, code_path):
    call(["python3", os.path.join(code_path, "run_loop_multi_cpu_just_org_image.py"), "-i",
          image_path, "-o", obj.output_path, "-if", obj.image_format])



def run_loop_multi_cpu_pec(obj, image_path, code_path):
    _, File = os.path.split(obj.Case)
    obj.File = File[:File.find(obj.air_seg_prefix)]

    call(["python3", os.path.join(code_path, "preprocessing_pec.py"),
          "-i", image_path, "-if", obj.image_format, "-cn", obj.File,
          "-po", obj.print_off, "-sfn", obj.saving_folder_name_net_pec,
          "-ar", str(obj.A_Range), "-fis", str(obj.final_image_size),
          "-o", obj.output_path])



def run_loop_multi_cpu_post(obj, image_path, code_path):
    _, File = os.path.split(obj.Case)
    obj.File = File[:File.find(obj.pec_seg_prefix)]

    call(["python3", os.path.join(code_path, "postprocessing.py"),
          "-i", image_path, "-if", obj.image_format, "-cn", obj.File,
          "-sfn", obj.saving_folder_name_final_masked_normalized_images,
          "-ar", str(obj.A_Range), "-fis", str(obj.final_image_size),
          "-o", obj.output_path, "-po", obj.print_off, "-fb", obj.find_bottom])



def run_loop_multi_cpu_denisty_map(obj, image_path, code_path):
    _, File = os.path.split(obj.Case)
    obj.File = File[:-4]

    call(["python3", os.path.join(obj.code_path, "density_map_feature_based.py"),
          "-i", obj.Case, "-if", obj.image_format, "-cn", obj.File,
          "-po", obj.print_off, "-sfn", obj.saving_folder_name_breast_density,
          "-ar", str(obj.A_Range), "-fis", str(obj.final_image_size),
          "-o", obj.output_path, "-lt", str(obj.libra_training),
          "-pttm", obj.model_path_density, "-rii", obj.remove_intermediate_images,
          "-to", str(obj.timeout_waiting)])



def get_the_image_reset_info(obj):
    Output_file_path_mask = os.path.join(obj.output_path, obj.File, "final_breast_mask", obj.File+"_final_mask"+obj.image_format)
    Output_file_path_image = os.path.join(obj.output_path, obj.File, obj.File+"_masked_image"+obj.image_format)
    csv_file = os.path.join(obj.output_path, obj.File, "air_breast_mask", "fixing_ratio.csv")

    image_reset_info = pd.read_csv(csv_file, sep=",", index_col=0)

    Image_needed_side_extention = image_reset_info["Image_needed_side_extention"].iloc[0]
    Image_Dimension_X = image_reset_info["Image_Dimension_X"].iloc[0]
    Image_Dimension_Y = image_reset_info["Image_Dimension_Y"].iloc[0]

    Max = max(Image_Dimension_X, Image_Dimension_Y)
    Mask = cv2.resize(obj.mask, (Max, Max), interpolation = cv2.INTER_NEAREST)

    if Image_needed_side_extention == "V":
        Mask = Mask[:, :Image_Dimension_Y]
    else:
        Mask = Mask[:Image_Dimension_X, :]

    Mask = np.uint8( Mask*255/Mask.max() )
    Mask = np.logical_not( find_largest_obj( np.logical_not(Mask>1) ) )
    cv2.imwrite(Output_file_path_mask, np.uint8(Mask*255))
    obj.mask = Mask>0


    Min_not_masked = obj.image[obj.mask].min()
    obj.image[np.logical_not(obj.mask)] = Min_not_masked
    obj.image = Normalize_Image(obj.image, obj.A_Range, bits_conversion=obj.bits_conversion)
    cv2.imwrite(Output_file_path_image)

    return(obj)
