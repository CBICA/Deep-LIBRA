import numpy as np
import os, pdb, cv2
import pandas as pd
from copy import deepcopy
from termcolor import colored
from keras.models import Model
import matplotlib.pyplot as plt

# My packages
from data_rw import testGenerator, get_image_info, saveResults_batch_based



def test_network_air(obj):
    obj.train_path = os.path.join(obj.output_path, obj.saving_folder_name_net_air)
    obj.saving_path = os.path.join(obj.output_path, obj.saving_folder_name_net_pec_temp)

    print("[INFO] The testing path is " + obj.train_path)
    print("[INFO] The saving path is " + obj.saving_path)
    print("[INFO] The model path is " + obj.model_path)

    if not(os.path.exists(obj.saving_path)): os.makedirs(obj.saving_path)

    if obj.print_off == "1":
        Verbose = 0
    else:
        Verbose = 1


    obj = get_image_info(obj)
    Test_set = testGenerator(obj)


    while True:
        indexes = next(Test_set.index_generator)
        images = Test_set._get_batches_of_transformed_samples(indexes)
        images = images/images.max()

        results = obj.model.predict(images, verbose=Verbose)
        image_names = []
        for index in indexes:
            image_names.append(Test_set.filenames[index])

        saveResults_batch_based(obj, results, image_names, obj.air_seg_prefix)

        if indexes[-1]==Test_set.n-1:
            break

    return(obj)


def test_network_pec(obj):
    obj.train_path = os.path.join(obj.output_path, obj.saving_folder_name_net_pec)
    obj.saving_path = os.path.join(obj.output_path, obj.saving_folder_name_temp_breast_masks)

    print("[INFO] The testing path is " + obj.train_path)
    print("[INFO] The saving path is " + obj.saving_path)
    print("[INFO] The model path is " + obj.model_path)

    if not(os.path.exists(obj.saving_path)): os.makedirs(obj.saving_path)

    if obj.print_off == "1":
        Verbose = 0
    else:
        Verbose = 1


    obj = get_image_info(obj)
    Test_set = testGenerator(obj)

    while True:
        indexes = next(Test_set.index_generator)
        images = Test_set._get_batches_of_transformed_samples(indexes)
        images = images/images.max()

        results = obj.model.predict(images, verbose=Verbose)
        image_names = []
        for index in indexes:
            image_names.append(Test_set.filenames[index])

        saveResults_batch_based(obj, results, image_names, obj.pec_seg_prefix)

        if indexes[-1]==Test_set.n-1:
            break

    return(obj)



def test_birads(obj):
    obj.train_path = os.path.join(obj.output_path, obj.saving_folder_name_final_masked_normalized_images)
    obj.saving_path = os.path.join(obj.output_path, obj.saving_folder_name_breast_density)


    if obj.print_off == "0":
        print(colored("[INFO]", "cyan") + " BIRADS assessment by network is started; please wait ...")
        print("[INFO] The testing path is " + obj.train_path)
        print("[INFO] The saving path is " + obj.saving_path)
        print("[INFO] The model path is " + obj.model_path)


    if not(os.path.exists(obj.saving_path)): os.makedirs(obj.saving_path)

    if obj.print_off == "1":
        Verbose = 0
    else:
        Verbose = 1


    obj = get_image_info(obj)
    Test_set = testGenerator(obj)

    # feature extractor model
    obj.Density_map_model = Model(obj.model.input, obj.model.layers[-6].output)


    Loop_counter = 0
    BIRADS_list = ["1", "2", "3", "4"]
    while True:
        indexes = next(Test_set.index_generator)
        images = Test_set._get_batches_of_transformed_samples(indexes)
        images = images/images.max()

        results = obj.model.predict(images, verbose=Verbose)
        features = obj.Density_map_model.predict(images, verbose=Verbose)
        features = features.reshape([len(indexes),-1])

        if len(indexes)==1:
            Image_BIRADS = pd.DataFrame(data=[results[0]],
                                        index=[Test_set.filenames[int(indexes.item())]],
                                        columns=BIRADS_list)
        else:
            Image_BIRADS = pd.DataFrame(data=results,
                                        index=Test_set.filenames[0:len(indexes)],
                                        columns=BIRADS_list)
        if Loop_counter == 0:
            BIRADS = deepcopy(Image_BIRADS)
        else:
            temp = [BIRADS, Image_BIRADS]
            BIRADS = pd.concat(temp)


        if len(indexes)==1:
            Image_features = pd.DataFrame(data=[features[0]],
                                        index=[Test_set.filenames[int(indexes.item())]])
        else:
            Image_features = pd.DataFrame(data=features,
                                        index=Test_set.filenames[0:len(indexes)])
        if Loop_counter == 0:
            Features = deepcopy(Image_features)
        else:
            temp = [Features, Image_features]
            Features = pd.concat(temp)


        Loop_counter += 1
        if indexes[-1]==Test_set.n-1:
            break


    BIRADS.to_csv(os.path.join(obj.saving_path, "Predicted_BIRADS_All_Images.csv"))
    Features.to_hdf(os.path.join(obj.saving_path,
                "Extracted_Features_All_images.h5"), key="Features", mode="w")
    return(obj)
