import warnings
warnings.filterwarnings("ignore")

#!/usr/bin/python3
from libra import *
from initialize_variables import set_argparse, get_variables


# python3 ~/github/LIBRA/final/execute_libra_code.py -i ~/comp_space/dataset/ -o ~/comp_space/dataset/libra_new2 -m ~/Net


class run_libra(object):
    def __init__(self):
        args = set_argparse(argv=None)
        self = get_variables(self, args)


    def main_function(self):
        Info = LIBRA()
        print(colored("[INFO] Starting LIBRA "+Info.version, 'green'))


        Info.parse_args(["-i", self.input_data,
                        "-po", self.print_off,
                         "-o", self.output_path,
                         "-ng", str(self.num_gpu),
                         "-mc", str(self.multi_cpu),
                         "-fb", str(self.find_bottom),
                         "-m", self.general_model_path,
                         "-lt", str(self.libra_training),
                         "-cm", str(self.core_multiplier),
                         "-fpm", str(self.find_pacemaker),
                         "-tow",str(self.timeout_waiting),
                         "-tbs", str(self.test_batch_size),
                         "-fis", str(self.final_image_size),
                         "-not", str(self.number_of_threads),
                         "-wsm", self.weight_selection_method,
                         "-wttbd", self.which_task_to_be_done,
                         "-rii", self.remove_intermediate_images,
                         "-lsm", str(self.libra_segmentation_method)])


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("a_pre")>-1 or \
            Info.which_task_to_be_done.find("a_cnn")>-1 or \
            Info.which_task_to_be_done.find("j_org")>-1:
            Info.get_info_based_on_air_cnn()


        if Info.which_task_to_be_done == "j_org":
            Info.run_just_orginal_image_preprocessing()


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("a_pre")>-1:
            Info.run_air_preprocessing()


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("a_cnn")>-1:
            Info.run_air_cnn()


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("p_pre")>-1 or \
            Info.which_task_to_be_done.find("p_cnn")>-1 or \
            Info.which_task_to_be_done.find("b_pos")>-1 or \
            Info.which_task_to_be_done.find("b_den")>-1:
            Info.get_info_based_on_pec_cnn()


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("p_pre")>-1:
            Info.run_pec_preprocessing()


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("p_cnn")>-1:
            Info.run_pec_cnn()


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("b_pos")>-1:
            Info.run_breast_postprocessing()


        if Info.which_task_to_be_done == "all" or \
            Info.which_task_to_be_done.find("b_den")>-1:
            Info.run_feature_extraction()


        T_End = time()
        print("[INFO] The total elapsed time (for all files): "+'\033[1m'+ \
              colored(str(round(T_End-Info.T_Start, 2)), 'red')+'\033[0m'+" seconds")
        print(colored("[INFO] *** The LIBRA is performed SUCCESSFULY and the results are SAVED ***", 'green'))



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################################### Running the code
if __name__ == "__main__":
    RUN = run_libra()
    RUN.main_function()
