import argparse, os


def set_argparse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dataset",
                    default='home/ohm/Desktop/data/train/',
                    help="Path to input dataset")

    ap.add_argument("-o", "--output_path",
                    default='home/ohm/Desktop/data/train/',
                    help="Path for saving results file")


    # models info
    ap.add_argument("-wsm", "--weight_selection_method", default="bv",
                    help="How to select the best weights")

    ap.add_argument("-m", "--general_model_path",
                    default="/home/ohm/Desktop/Net",
                    help="A general path to where the models are saved")

    ap.add_argument("-ma", "--model_path_air",
                    default="network_model.h5",
                    help="The name of the saved and trained air_pec_breast model")

    ap.add_argument("-mp", "--model_path_pec",
                    default="network_model.h5",
                    help="The name of the saved and trained pec_breast model")

    ap.add_argument("-md", "--model_path_density",
                    default="network_model.pkl",
                    help="The name of the saved and trained breast density model")


    # all images to be saved in; you can go with defaults
    ap.add_argument("-sfnna", "--saving_folder_name_net_air",
                    default="air_net_data",
                    help="Foldername for saving the preprocessed air results")

    ap.add_argument("-sfnnp", "--saving_folder_name_net_pec",
                    default="pec_net_data",
                    help="Foldername for saving preprocessing pectoral results")

    ap.add_argument("-sfntbm", "--saving_folder_name_temp_breast_masks",
                    default="breast_temp_masks",
                    help="Foldername for saving temp resutls out of the pectoral cnn")

    ap.add_argument("-sfnfni", "--saving_folder_name_final_masked_normalized_images",
                    default="final_images",
                    help="Foldername to final masked and normalized images")

    ap.add_argument("-sfnbd", "--saving_folder_name_breast_density",
                    default="breast_density",
                    help="Foldernam to breast density results")


    # GPU CPU conditions
    ap.add_argument("-mc", "--multi_cpu", type=int, default=0,
                    help="If you want to use maximum power of PC using multi core CPUs, ."+
                    "should be one. The defualt (zero) is using just one core.")

    ap.add_argument("-not", "--number_of_threads", type=int, default=10,
                    help="How many threads for each CPU core to be considered.")

    ap.add_argument("-cm", "--core_multiplier", type=int, default=4,
                    help="How many batches to be open to wait for clsoign Queue.")

    ap.add_argument("-ng", "--num_gpu", type=int, default=1,
                    help="Number of GPU for being used in training. 0 means run by CPU.")

    ap.add_argument("-tbs", "--test_batch_size", type=int, default=10,
                    help="The number of images in test batch size.")

    ap.add_argument("-tow", "--timeout_waiting", type=int, default=180,
                    help="Timeout waiting value that if the time exceed than this "+
                    "number the tasks will break for density map generation step. "+
                    "The default is 3 minutes for each job.")


    # Other parametersget_network_segmentation
    ap.add_argument("-lt", "--libra_training", type=int, default=0,
                    help="If one, then, its for training.")

    ap.add_argument("-fb", "--find_bottom", default="1",
                    help="if this is one, it tries to remove the bottom.")

    ap.add_argument("-fpm", "--find_pacemaker", default=0,
                    help="If this is one, it will remove the pacemakers by replacing it with minimum.")

    ap.add_argument("-lsm", "--libra_segmentation_method", default="Libra",
                    help="It can be Libra or Exaturated.")

    ap.add_argument("-po", "--print_off", default="1",
                    help="I just limits the printing to log if it si one; this is needed for batch processing.")

    ap.add_argument("-fis", "--final_image_size", type=int, default=512,
                    help="This number should be selected based on the trained network. Keep it constant!")


    # if you want to use just one specific part of the method
    ap.add_argument("-wttbd", "--which_task_to_be_done",
                    default="all",
                    help="This is a flag to show which task/s to be performed. It is really useful " +
                    "in training or for running specific part. The options are: " +
                    "all, a_a (after preprocessing_air air), a_c_a (after cnn air), " +
                    "a_p (after pectoral preprocessing), a_c_p (after pectoral cnn), " +
                    "j_s (just segmentation)")

    ap.add_argument("-rii", "--remove_intermediate_images",
                    default="K", help="R is removing and K is keeping them")

    args = vars(ap.parse_args(argv))

    return(args)



def get_variables(obj, args):
    obj.output_path = args["output_path"]
    obj.input_data = args["input_dataset"]


    obj.general_model_path = args["general_model_path"]
    obj.model_path_air = args["model_path_air"]
    obj.model_path_air = os.path.join(obj.general_model_path, "air", obj.model_path_air)
    obj.model_path_pec = args["model_path_pec"]
    obj.model_path_pec = os.path.join(obj.general_model_path, "pectoral", obj.model_path_pec)
    obj.model_path_density = args["model_path_density"]
    obj.model_path_density = os.path.join(obj.general_model_path, "density", obj.model_path_density)


    obj.saving_folder_name_net_air = args["saving_folder_name_net_air"]
    obj.saving_folder_name_net_air = os.path.join(obj.saving_folder_name_net_air, "image")
    obj.saving_folder_name_net_pec = args["saving_folder_name_net_pec"]
    obj.saving_folder_name_net_pec = os.path.join(obj.saving_folder_name_net_pec, "image")
    obj.saving_folder_name_breast_density = args["saving_folder_name_breast_density"]
    obj.saving_folder_name_temp_breast_masks = args["saving_folder_name_temp_breast_masks"]
    obj.saving_folder_name_final_masked_normalized_images = args["saving_folder_name_final_masked_normalized_images"]
    obj.saving_folder_name_final_masked_normalized_images = os.path.join(obj.saving_folder_name_final_masked_normalized_images, "image")


    obj.air_seg_prefix = "_final_air_predict"
    obj.pec_seg_prefix = "_final_pec_predict"


    obj.num_gpu = args["num_gpu"]
    obj.print_off = args["print_off"]
    obj.find_bottom = args["find_bottom"]
    obj.find_pacemaker = args["find_pacemaker"]
    obj.test_batch_size = args["test_batch_size"]
    obj.final_image_size = args["final_image_size"]
    obj.libra_training = int(args["libra_training"])
    obj.which_task_to_be_done = args["which_task_to_be_done"]
    obj.weight_selection_method = args["weight_selection_method"]
    obj.libra_segmentation_method = args["libra_segmentation_method"]
    obj.remove_intermediate_images = args["remove_intermediate_images"]


    obj.timeout_waiting = args["timeout_waiting"]

    obj.multi_cpu = args["multi_cpu"]
    obj.core_multiplier = args["core_multiplier"]
    obj.number_of_threads = args["number_of_threads"]

    return(obj)
