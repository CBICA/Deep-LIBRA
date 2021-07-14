import pdb, six
import numpy as np
import pandas as pd
import SimpleITK as sitk
from copy import deepcopy
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from radiomics import firstorder, glcm, ngtdm, gldm, glrlm, glszm


# turn off logging notices
import logging
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)



def extract_radiomics_features(obj, image, mask=[]):
    if len(mask)==0:
        mask = np.zeros(image.shape)

    try:
        bits_conversion = obj.bits_conversion
    except:
        bits_conversion = "uint8"


    features = []
    features_name = []

    List_featrues = ["avg", "std", "ske", "kur"]
    LBPParams = [{"radius":1, "points":8}, {"radius":3, "points":24}]
    for Param in LBPParams:
        for F in List_featrues:
            features_name.append("LBP_"+ F +"_R" + str(Param["radius"])
                +"_P"+str(Param["points"]))

    # LBP features maps
    METHOD = 'uniform'
    LBP_filtered_Images = []
    for LBP in LBPParams:
        lbp = local_binary_pattern(image, LBP["points"], LBP["radius"], METHOD)
        LBP_filtered_Images.append(lbp)

    for LBPI in LBP_filtered_Images:
        features.append(LBPI[mask>0].mean())
        features.append(LBPI[mask>0].std())
        features.append(skew(LBPI[mask>0]))
        features.append(kurtosis(LBPI[mask>0]))


    if image.shape[-1]==3:
        image = np.reshape(image, (3, image.shape[0], image.shape[1]))
        mask = np.reshape(mask, (3, mask.shape[0], image.shape[1]))
    else:
        image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        mask = np.reshape(mask, (1, mask.shape[0], image.shape[1]))
    image = image.astype(bits_conversion)



    mask = mask.astype(bits_conversion)


    Image_ITK = sitk.GetImageFromArray(image)
    Mask_ITK = sitk.GetImageFromArray(mask)


    List_features = ["firstorder", "glcm", "ngtdm", "gldm", "glrlm", "glszm"]
    List_radiomics_labels = ["RadiomicsFirstOrder", "RadiomicsGLCM",
                             "RadiomicsNGTDM", "RadiomicsGLDM", "RadiomicsGLRLM", "RadiomicsGLSZM"]
    for feature, function_to_load in zip(List_features, List_radiomics_labels):
        code = feature+"."+function_to_load+"(Image_ITK, Mask_ITK)"
        active_feature=eval(code)
        exec('active_feature.enableAllFeatures()')
        exec('active_feature.execute()')
        # pdb.set_trace()
        for (key, val) in six.iteritems(eval('active_feature.featureValues')):
            features.append(val.item())
            features_name.append(key)

    obj.FEATUREs = pd.DataFrame(data=[features], columns=features_name)

    return(obj)



################################################################################
################################################################################
################################################################################
################################################################################
def set_slic_features(image, mask, segments, Image_ITK, itk_mask, itk_segments,
                      seg, LBP_filtered_Images, features, features_names, counter):
    seg_features = deepcopy(features)
    List_features = ["firstorder", "glcm"]
    List_radiomics_labels = ["RadiomicsFirstOrder", "RadiomicsGLCM"]

    if counter==0:
        List_featrues = ["avg", "std", "ske", "kur"]
        LBPParams = [{"radius":1, "points":8}, {"radius":3, "points":24}]
        for Param in LBPParams:
            for F in List_featrues:
                features_names.append("Seg_LBP_"+ F +"_R" + str(Param["radius"])
                    +"_P"+str(Param["points"]))

        seg_mask = np.zeros(mask.shape)
        seg_mask[segments==seg] = 1
        for LBPI in LBP_filtered_Images:
            seg_features.append(LBPI[seg_mask>0].mean())
            seg_features.append(LBPI[seg_mask>0].std())
            seg_features.append(skew(LBPI[seg_mask>0]))
            seg_features.append(kurtosis(LBPI[seg_mask>0]))


        seg_mask = np.zeros(itk_mask.shape)
        seg_mask[itk_segments==seg]=1
        Mask_ITK = sitk.GetImageFromArray(seg_mask)
        for feature, function_to_load in zip(List_features, List_radiomics_labels):
            code = feature+"."+function_to_load+"(Image_ITK, Mask_ITK)"
            active_feature=eval(code)
            exec('active_feature.enableAllFeatures()')
            exec('active_feature.execute()')

            for (key, val) in six.iteritems(eval('active_feature.featureValues')):
                seg_features.append(val.item())
                features_names.append("Seg_"+key)

        features_names.append("Seg_area")
        features_names.append("Seg_index")


    else:
        seg_mask = np.zeros(mask.shape)
        seg_mask[segments==seg] = 1
        for LBPI in LBP_filtered_Images:
            seg_features.append(LBPI[seg_mask>0].mean())
            seg_features.append(LBPI[seg_mask>0].std())
            seg_features.append(skew(LBPI[seg_mask>0]))
            seg_features.append(kurtosis(LBPI[seg_mask>0]))


        seg_mask = np.zeros(itk_mask.shape)
        seg_mask[itk_segments==seg]=1
        Mask_ITK = sitk.GetImageFromArray(seg_mask)
        for feature, function_to_load in zip(List_features, List_radiomics_labels):
            code = feature+"."+function_to_load+"(Image_ITK, Mask_ITK)"
            active_feature=eval(code)
            exec('active_feature.enableAllFeatures()')
            exec('active_feature.execute()')

            for (key, val) in six.iteritems(eval('active_feature.featureValues')):
                seg_features.append(val.item())

    Seg_area = np.sum(np.logical_and(segments==seg, mask>0))
    seg_features.append(Seg_area)

    seg_features.append(seg)

    return(seg_features, features_names)






################################################################################
def extract_breast_radiomics_features(obj, image, mask=[], segments=[],
            case_name=np.nan, Minimum_acceptable_number_of_pixels_in_segment=49):
    if len(mask)==0:
        mask = np.zeros(image.shape)

    try:
        bits_conversion = obj.bits_conversion
    except:
        bits_conversion = "uint8"

    # image = image.astype(bits_conversion)
    mask = mask.astype(bits_conversion)


    features = []
    features_names = []


    Breast_area_total = np.sum(mask>0)
    features.append(Breast_area_total)
    features_names.append("Breast_area")


    List_featrues = ["avg", "std", "ske", "kur"]
    LBPParams = [{"radius":1, "points":8}, {"radius":3, "points":24}]
    for Param in LBPParams:
        for F in List_featrues:
            features_names.append("LBP_"+ F +"_R" + str(Param["radius"])
                +"_P"+str(Param["points"]))

    # LBP features maps
    METHOD = 'uniform'
    LBP_filtered_Images = []
    for LBP in LBPParams:
        lbp = local_binary_pattern(image, LBP["points"], LBP["radius"], METHOD)
        LBP_filtered_Images.append(lbp)

    for LBPI in LBP_filtered_Images:
        features.append(LBPI[mask>0].mean())
        features.append(LBPI[mask>0].std())
        features.append(skew(LBPI[mask>0]))
        features.append(kurtosis(LBPI[mask>0]))


    if image.shape[-1]==3:
        itk_image = np.reshape(image, (3, image.shape[0], image.shape[1]))
        itk_mask = np.reshape(mask, (3, mask.shape[0], image.shape[1]))
        itk_segments = np.reshape(segments, (3, segments.shape[0], segments.shape[1]))
    else:
        itk_image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        itk_mask = np.reshape(mask, (1, mask.shape[0], image.shape[1]))
        itk_segments = np.reshape(segments, (1, segments.shape[0], segments.shape[1]))


    Image_ITK = sitk.GetImageFromArray(itk_image)
    Mask_ITK = sitk.GetImageFromArray(itk_mask)

    List_features = ["firstorder", "glcm", "ngtdm", "gldm", "glrlm", "glszm"]
    List_radiomics_labels = ["RadiomicsFirstOrder", "RadiomicsGLCM",
                             "RadiomicsNGTDM", "RadiomicsGLDM", "RadiomicsGLRLM", "RadiomicsGLSZM"]
    for feature, function_to_load in zip(List_features, List_radiomics_labels):
        code = feature+"."+function_to_load+"(Image_ITK, Mask_ITK)"
        active_feature=eval(code)
        exec('active_feature.enableAllFeatures()')
        exec('active_feature.execute()')

        for (key, val) in six.iteritems(eval('active_feature.featureValues')):
            features.append(val.item())
            features_names.append(key)

    N = 0
    for seg in np.unique(itk_segments[itk_mask>0]):
        if (np.sum(np.logical_and(segments==seg, mask>0))>Minimum_acceptable_number_of_pixels_in_segment):
            condition_to_remove = np.logical_and(segments==seg, np.logical_not(mask>0))
            if condition_to_remove.any():
                segments[condition_to_remove] = -1
            segments[np.logical_and(segments==seg, np.logical_not(mask>0))] = -1
            seg_features, features_names = set_slic_features(image, mask, segments,
                                Image_ITK, itk_mask, itk_segments, seg, LBP_filtered_Images,
                                features, features_names, N)

            if N==0:
                obj.FEATUREs = pd.DataFrame(data=[seg_features], columns=features_names)
            else:
                temp = pd.DataFrame(data=[seg_features], columns=features_names)
                obj.FEATUREs = [obj.FEATUREs, temp]
                obj.FEATUREs = pd.concat(obj.FEATUREs, ignore_index=True)

            N += 1

    return(obj, segments)
