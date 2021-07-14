# This is read and write functions needed for deeplearning
import pdb, os
import numpy as np
from termcolor import colored

# it finds the best weight for 20 percent of epochs and after
def weight_selection(obj, parameter_for_selection, final_index=-1, max_index=-1):
    if obj.weight_selection_method != "NA":
        val_param = "val_"+parameter_for_selection
        tra_param = parameter_for_selection

        val_param = np.array(obj.json[val_param])
        tra_param = np.array(obj.json[tra_param])
        if max_index!=-1:
            val_param = val_param[:max_index]
            tra_param = tra_param[:max_index]

        if obj.weight_selection_method == "bvtw": # best val and train (train(50%))


            final_param = (2*val_param+tra_param)/3
            final_param[:int(0.2*final_param.shape[0])] = 0

            max_index = np.argmax(final_param[::obj.save_period])*obj.save_period


        elif obj.weight_selection_method == "bvt": # best val and train (train(50%))
            val_param = np.array(obj.json[val_param])
            tra_param = np.array(obj.json[tra_param])

            final_param = (val_param+tra_param)/2
            final_param[:int(0.2*final_param.shape[0])] = 0

            max_index = np.argmax(final_param[::obj.save_period])*obj.save_period


        elif obj.weight_selection_method == "bv": # best val
            val_param[:int(0.2*val_param.shape[0])] = 0
            max_index = np.argmax(val_param[::obj.save_period])*obj.save_period


        elif obj.weight_selection_method == "bt": # best training
            tra_param[:int(0.2*tra_param.shape[0])] = 0
            max_index = np.argmax(tra_param[::obj.save_period])*obj.save_period


        if final_index==-1:
            final_index = max_index

        Path, net_name = os.path.split(obj.model_path)
        if net_name.find("_model.h5")>-1:
            weight_name = net_name[:net_name.find("_model.h5")]
        else:
            weight_name = net_name[:net_name.find("_Model.h5")]
        weight_name = weight_name+"_weights_M_%08d.h5" % final_index

        print(colored("The best model is: "+weight_name % final_index, 'yellow'))

        obj.model.load_weights( os.path.join(Path, weight_name) )

        print("The validation "+parameter_for_selection+" was "+
              colored(str(round(val_param[final_index], 4)),'red')+
              " and the training "+parameter_for_selection+" was "+
              colored(str(round(tra_param[final_index], 4)),'red'))

    return(obj)
