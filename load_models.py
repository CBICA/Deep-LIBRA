import os, pdb
from keras.models import load_model
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
# other packages
from get_info import get_info_from_network
from weight_selection import weight_selection
# the metrics
from metrics import sensitivity, sensitivity_weighted
from metrics import dice_weighted, loss_dice_weighted, dice
from metrics import generalised_dice, generalised_dice_loss, loss_dice
from metrics import dice_weighted_traditional, loss_dice_weighted_traditional
from metrics import generalised_wasserstein_dice, generalised_wasserstein_dice_loss, generalised_wasserstein_dice_loss2
# new metrics and losses
from seg_metrics import Dice, IOU
from seg_losses import jaccard_loss, bce_jaccard_loss, dice_loss, bce_dice_loss



def get_network_segmentation(obj, model_path, Keys_input=[], Keys_output=[], max_index=-1):
    obj = get_info_from_network(obj, model_path, Keys_input, Keys_output)

    if obj.training_mode == "3_Class_Breast":
        obj.model = load_model(model_path, custom_objects={"dice_weighted": dice_weighted,
                    "loss_dice_weighted": loss_dice_weighted, "sensitivity": sensitivity,
                    "sensitivity_weighted": sensitivity_weighted, "dice":dice})
        obj = weight_selection(obj, "dice_weighted", max_index)

    elif obj.training_mode == "3_Class_Breast_N":
        obj.model = load_model(model_path, custom_objects={"dice_weighted": dice_weighted,
                    "loss_dice_weighted": loss_dice_weighted, "sensitivity": sensitivity,
                    "sensitivity_weighted": sensitivity_weighted, "dice":dice,
                    "dice_weighted_traditional": dice_weighted_traditional})
        obj = weight_selection(obj, "dice_weighted", max_index)

    elif obj.training_mode == "3_Class_Breast_Traditional":
        obj.model = load_model(model_path, custom_objects={"dice_weighted_traditional": dice_weighted_traditional,
                    "loss_dice_weighted_traditional": loss_dice_weighted_traditional, "sensitivity": sensitivity,
                    "sensitivity_weighted": sensitivity_weighted, "dice":dice, "dice_weighted": dice_weighted})
        obj = weight_selection(obj, "dice_weighted_traditional", max_index)

    elif obj.training_mode == "N_Class_General":
        obj.model = load_model(model_path, custom_objects={"loss_dice": loss_dice, "sensitivity": sensitivity,
                    "dice":dice})
        obj = weight_selection(obj, "dice", max_index)

    elif obj.training_mode == "N_Class_Generalize_2017":
        obj.model = load_model(model_path, custom_objects={"generalised_dice": generalised_dice,
                    "generalised_dice_loss": generalised_dice_loss, "sensitivity": sensitivity,
                    "dice":dice})
        obj = weight_selection(obj, "generalised_dice", max_index)

    elif obj.training_mode == "Categorical_loss":
        obj.model = load_model(model_path, custom_objects={"categorical_accuracy": categorical_accuracy,
                    "categorical_crossentropy": categorical_crossentropy, "sensitivity": sensitivity,
                    "dice":dice})
        obj = weight_selection(obj, "categorical_accuracy", max_index)

    elif obj.training_mode == "jaccard_loss":
        obj.model = load_model(model_path, custom_objects={"jaccard_loss": jaccard_loss,
                    "Dice": Dice, "IOU": IOU, "sensitivity":sensitivity})
        obj = weight_selection(obj, "IOU", max_index)

    elif obj.training_mode == "dice_loss":
        obj.model = load_model(model_path, custom_objects={"dice_loss": dice_loss,
                    "Dice": Dice, "IOU": IOU, "sensitivity":sensitivity})
        obj = weight_selection(obj, "Dice", max_index)

    elif obj.training_mode == "wasserstein_3_Class_Breast":
        obj.model = load_model(model_path, custom_objects={"generalised_wasserstein_dice": generalised_wasserstein_dice,
                    "generalised_wasserstein_dice_loss": generalised_wasserstein_dice_loss, "sensitivity": sensitivity,
                    "dice_weighted": dice_weighted, "dice":dice,
                    "sensitivity_weighted": sensitivity_weighted})
        obj = weight_selection(obj, "generalised_wasserstein_dice", max_index)

    elif obj.training_mode == "wasserstein":
        obj.model = load_model(model_path, custom_objects={"generalised_wasserstein_dice": generalised_wasserstein_dice,
                    "generalised_wasserstein_dice_loss2": generalised_wasserstein_dice_loss2, "sensitivity": sensitivity,
                    "dice":dice, "sensitivity_weighted": sensitivity_weighted})
        obj = weight_selection(obj, "generalised_wasserstein_dice", max_index)

    return(obj)




def get_network_classification(obj, model_path, Keys_input, Keys_output, max_index=-1):
    obj = get_info_from_network(obj, model_path, Keys_input, Keys_output, default_values=None)

    obj.model = load_model(model_path, custom_objects={"categorical_crossentropy": categorical_crossentropy})
    obj = weight_selection(obj, "acc", max_index)

    return(obj)
