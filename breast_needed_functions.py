import numpy as np
import pandas as pd
import cv2, pdb, os
from scipy import signal
from copy import deepcopy
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d


################################################################################
################################################################################
def air_Libra(obj):
    # LIBRA method for raw images
    row_range = obj.image.max(axis=1)-obj.image.min(axis=1)
    row_range = row_range/row_range.max()

    C1 = np.where(row_range>0.001)[0][0]
    C2 = np.where(row_range>0.001)[0][-1]
    I0_center = obj.image[C1:C2, :]

    img_pixs = I0_center.reshape([1,-1])
    x = np.arange(img_pixs.min(), img_pixs.max(),
                  (img_pixs.max()-img_pixs.min())/1000.0 )
    n_elements = np.histogram(img_pixs, bins=1000, range=(img_pixs.min(),
                                                          img_pixs.max()))
    c_elements = np.insert(np.cumsum(n_elements[0]),[0], [0])
    dd = np.diff(c_elements)
    dd = gaussian_filter1d(dd, sigma=1)
    peaklocation = np.where(dd>dd.max()*0.05)[0][0]


    ddd = dd[1:] - dd[:-1]
    ddd_neg = peaklocation + np.where(ddd[peaklocation+1:]<=0)[0][0]
    threhsold_opt1 = ddd_neg + np.where(ddd[ddd_neg+1:]>=0)[0][0]-4
    threhsold = min(x[threhsold_opt1], x[peaklocation+30])


    mask = obj.image <= threhsold

    if hasattr(obj, "Shrinking_ratio"):
        mask = cv2.resize(mask.astype("int")*1.0,
                          (round(obj.image.shape[1]/obj.Shrinking_ratio),
                           round(obj.image.shape[0]/obj.Shrinking_ratio)),
                          interpolation = cv2.INTER_AREA)
        mask = mask>0

    return(mask)


################################################################################
################################################################################
def air(obj):
    # # OTSU method for raw images
    threhsold = threshold_otsu(obj.img_norm)
    mask = obj.img_norm <= threhsold
    return(mask)



################################################################################
################################################################################
def find_logical_background_objs(Mask):
    # the background should be zero and object one
    temp_mask = Mask.astype("int")

    temp_mask[0,:] = 0
    labeled_obj = label(temp_mask)[0]
    BG_Label = labeled_obj[0,-1]

    if labeled_obj.max()>1:
        for num in range(labeled_obj.max()+1):
            if num != BG_Label:
                Loc = np.where(labeled_obj==num)
                if not( Loc[0].min()==0 or Loc[0].max()==Mask.shape[0] or
                    Loc[1].min()==0 or Loc[1].max()==Mask.shape[1] ):
                    Mask[labeled_obj==num] = True

    return(Mask)



################################################################################
################################################################################
def find_logical_pec_objs(Mask):
    # the pectoral should be one and rests zero (background called here)
    temp_mask = Mask.astype("int")

    temp_mask[0,:] = 0
    labeled_obj = label(temp_mask)[0]
    BG_Label = labeled_obj[0, -1]

    if labeled_obj.max()>1:
        for num in range(labeled_obj.max()+1):
            if num != BG_Label:
                Loc = np.where(labeled_obj==num)
                if not( Loc[0].min()==0 or Loc[1].min()==0 ):
                    Mask[labeled_obj==num] = False

    return(Mask)



################################################################################
################################################################################
def find_largest_obj(Mask):
    # zero shows background and one is objects
    temp_mask = Mask.astype("int")
    out_mask = deepcopy(Mask)

    # make the first row zero to make sure it is not affected by noise
    temp_mask[0,:] = 0

    labeled_obj = label(temp_mask)[0]

    if labeled_obj.max()>1:
        BG_Label = labeled_obj[0, -1]
        Unique_labels, counts = np.unique(labeled_obj, return_counts=True)
        counts = np.delete(counts, np.where(Unique_labels==BG_Label), None)
        Unique_labels = np.delete(Unique_labels, np.where(Unique_labels==BG_Label), None)
        Max = Unique_labels[counts.argmax()]
        out_mask[labeled_obj!=Max] = False

    return(out_mask)



################################################################################
################################################################################
def detect_buttom_portion(obj, Mask):
    mask = deepcopy(Mask)
    mask[:int(mask.shape[0]/2), :] = 0
    non_zero_index = np.argwhere(mask>0)
    # careful X is to bottom and y is to right
    y_direction_indexes = non_zero_index[:,1]
    x_direction_indexes = non_zero_index[:,0]

    y_max_indexes = np.argwhere(y_direction_indexes==y_direction_indexes.max(axis=0))
    x_max_y_max_index = x_direction_indexes[y_max_indexes].max()

    # remove anything above 1.3max_x
    mask[:min(int(1.3*x_max_y_max_index), mask.shape[0]), :] = 0

    if (mask>0).any():
        # Now repeat it to be Safe
        non_zero_index = np.argwhere(mask>0)
        # careful X is to bottom and y is to right
        y_direction_indexes = non_zero_index[:,1]
        x_direction_indexes = non_zero_index[:,0]

        y_max_indexes = np.argwhere(y_direction_indexes==y_direction_indexes.max(axis=0))
        x_max_y_max_index = x_direction_indexes[y_max_indexes].max()

        # remove anything above max_x
        mask[:x_max_y_max_index, :] = 0


    if (mask>0).any():
        contours, hierarchy = cv2.findContours((mask*255).astype("uint8"),
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # make sure taht there is just one contour
        if len(contours)==1:
            contour = contours[0].reshape([-1,2])

            # careful X is to bottom and y is to right
            Y_coordinates = contour[:,0]
            X_coordinates = contour[:,1]

            Indexes_touching_top_Xs = np.argwhere(X_coordinates==x_max_y_max_index)
            Indexe_max_Y = Y_coordinates[Indexes_touching_top_Xs].argmax()
            Index_top_right_corner = Indexes_touching_top_Xs[Indexe_max_Y]

            Indexes_touching_bottom_Xs = np.argwhere(X_coordinates==X_coordinates.max())
            if len(Indexes_touching_bottom_Xs) == 1:
                Index_bottom_right_corner = Indexes_touching_bottom_Xs[0]
            else:
                Indexe_max_Y = Y_coordinates[Indexes_touching_bottom_Xs].argmax()
                Index_bottom_right_corner = Indexes_touching_bottom_Xs[Indexe_max_Y]


            contour = contour[int(min(Index_top_right_corner,Index_bottom_right_corner)):
                            int(max(Index_top_right_corner,Index_bottom_right_corner)),:]

            if len(contour)!=0:
                # sort the coodrinates if needed
                if contour[0,1]>contour[-1,1]:
                    contour = contour[::-1]

                # careful X is to bottom and y is to right
                Y_coordinates = contour[:,0]
                X_coordinates = contour[:,1]

                b, a = signal.butter(5, 0.05, 'low')
                try:
                    Y_coordinates = signal.filtfilt(b, a, Y_coordinates)
                except:
                    Y_coordinates
                Y_diff = np.diff(Y_coordinates)

                try:
                    X_coordinates = signal.filtfilt(b, a, X_coordinates)
                except:
                    X_coordinates
                X_diff = np.diff(X_coordinates)

                Threshold_for_changes = 0.1
                Threshold_for_come_back = 0.5

                # turn back for breast
                if (X_diff<-Threshold_for_come_back).any():
                        first_index = np.where(X_diff<-Threshold_for_come_back)[0][0]
                        cut_diff = X_diff[first_index:]

                        # larger threshold for come back if needed
                        removing_index = np.where(cut_diff>Threshold_for_come_back)[0][0]
                        contour[removing_index+first_index,:]
                        Mask[:contour[removing_index+first_index,:][0],
                             contour[removing_index+first_index,:][1]:]=0


                elif (Y_diff>Threshold_for_changes).any():
                        first_index = np.where(Y_diff>Threshold_for_changes)[0][0]
                        removing_loc = contour[first_index+2,:]

                        range_to_remove_x = np.arange(removing_loc[1], contour[first_index:,1].max()+1)
                        range_to_remove_y = np.interp(range_to_remove_x, contour[first_index:,1],
                                                      contour[first_index:,0]).astype(int)+2

                        removing_indexes = np.concatenate((range_to_remove_x.reshape([-1,1]),
                                        range_to_remove_y.reshape([-1,1])), axis=1)

                        for removing_index in removing_indexes:
                            Mask[removing_index[0], :removing_index[1]] = 0


        Mask = Mask>0

    return(Mask)

################################################################################
################################################################################
def Normalize_Image(IMAGE, Range, Min=None, Max=None, flag_max_edition=None,
                    flag_min_edition=None, bits_conversion=None, Name=None):
    IMAGE = IMAGE.astype('float')

    if Min==None: Min = IMAGE.min()
    if Max==None: Max = IMAGE.max()

    if Min != Max:
        Out_Img = (IMAGE-Min)/(Max-Min)

        Out_Img = Out_Img*Range

        if flag_max_edition == None:
            try:
                Out_Img[Out_Img>Range] = Range
            except:
                Out_Img = Out_Img

        if flag_min_edition == None:
            try:
                Out_Img[Out_Img<0] = 0
            except:
                Out_Img = Out_Img

        if bits_conversion == None:
            if Range == 2**16-1:
                Out_Img = Out_Img.astype('uint16')
            else:
                Out_Img = Out_Img.astype('uint8')
        else:
            Out_Img = Out_Img.astype(bits_conversion)

    else:
        if Name == None:
            print("ERROR: SOMTHING WENT WRONG")
        else:
            print("ERROR: SOMTHING WENT WRONG for "+Name)

    return(Out_Img)



################################################################################
################################################################################
def get_headers(ds, List):
    values = []
    for item in List:
        if hasattr(ds, item):
            temp = getattr(ds, item)
            try:
                if item == "PatientAge" and temp[-1]=="Y":
                    temp = temp[:-1]
                try:
                    values.append(int(temp))
                except:
                    values.append(temp)
            except:
                values.append(np.nan)
        else:
            values.append(np.nan)

    Data = pd.DataFrame(data=[values], index=[0], columns=List)

    return(Data)



################################################################################
################################################################################
def object_oriented_preprocessing(obj, metal_threshold=30000,
                                  max_image_threshold=64000):
    # max_image_threshold never meets! just in case. metal_threshold is useful
    image = deepcopy(obj.image)
    image_metal = deepcopy(image)

    if (hasattr(obj, "find_pacemaker") and obj.find_pacemaker==1
        and obj.ds.PhotometricInterpretation!='MONOCHROME1'):
        image_metal[image_metal>max_image_threshold]=max_image_threshold
        if obj.find_pacemaker==1:
            MIN = image_metal.min()
            if (image_metal>metal_threshold).any():
                image_metal[image_metal>metal_threshold]=MIN

    if obj.ds.PresentationIntentType=='FOR PROCESSING':
        if image.min() < 1:
            image = image + abs(image.min()) + 1
            image_metal = image_metal + abs(image_metal.min()) + 1
        image = np.log(image)
        image_metal = np.log(image_metal)

    if obj.ds.PhotometricInterpretation=='MONOCHROME1':
        image = abs(image-image.max())
        image_metal = abs(image_metal-image_metal.max())

        if hasattr(obj, "find_pacemaker") and obj.find_pacemaker==1:
            image_metal = np.exp(image_metal)
            MIN = image_metal.min()
            image_metal[image_metal>metal_threshold]=MIN
            image_metal = np.log(image_metal)

    if obj.ds.PresentationIntentType=='FOR PROCESSING':
        image = image**2
        image_metal = image_metal**2


    if not(hasattr(obj.ds,'ImageLaterality')) and hasattr(obj.ds,'Laterality'):
        obj.ds.ImageLaterality = obj.ds.Laterality
    elif not(hasattr(obj.ds,'ImageLaterality')) and not(hasattr(obj.ds,'Laterality')):
        left = image[:, :int(image.shape[1]/2)].sum()
        right = image[:, int(image.shape[1]/2):].sum()
        if right>left:
            obj.ds.ImageLaterality = "R"
        else:
            obj.ds.ImageLaterality = "L"


    obj.fliping_flag = 0
    if hasattr(obj.ds,'FieldOfViewHorizontalFlip') and obj.ds.FieldOfViewHorizontalFlip =='YES':
        if obj.ds.ImageLaterality == 'L':
            image=np.fliplr(image)
            image_metal=np.fliplr(image_metal)
            obj.fliping_flag = 1
    else:
        if obj.ds.ImageLaterality == 'R':
            image=np.fliplr(image)
            image_metal=np.fliplr(image_metal)
            obj.fliping_flag = 1

    obj.image = image

    return (obj, image_metal)



################################################################################
################################################################################
def Remove_Top_Below_Side_effect(obj):
    MIN = obj.image.min()
    # Top
    for n, row in enumerate(obj.image):
        if not( (row==MIN).all() or (row==row[0]).all()):
            obj.top_n = n
            break

    # Bottom
    for n, row in enumerate(reversed(obj.image)):
        if not( (row==MIN).all() or (row==row[0]).all()):
            obj.bottom_n = n
            break

    for n, cols in enumerate(reversed(obj.image.T)):
        if not( (row==MIN).all() ):
            obj.side_n = n
            break

    if obj.bottom_n == 0:
        bottom_n = 1
    else:
        bottom_n = obj.bottom_n

    if obj.side_n == 0:
        side_n = 1
    else:
        side_n = obj.side_n

    Non_min_image = obj.image[obj.top_n:-bottom_n, :-side_n]
    MIN = Non_min_image.min()

    if obj.top_n>0:
        obj.image[:obj.top_n,:] = MIN

    if obj.bottom_n>0:
        obj.image[-obj.bottom_n:,:] = MIN

    if obj.side_n>0:
        obj.image[:,:-obj.side_n] = MIN

    return(obj)



################################################################################
################################################################################
def fix_ratio(IMG, height, width, method="area"):
    Flag = 0
    if np.array_equal(IMG, IMG.astype(bool)):
        IMG = IMG.astype("uint8")
        Flag = 1

    MIN = IMG.min()

    if IMG.shape[0] > IMG.shape[1]:
        IMG = np.concatenate((IMG, np.ones([IMG.shape[0],
                            IMG.shape[0]-IMG.shape[1]])*MIN), axis=1)
    else:
        IMG = np.concatenate((IMG, np.ones([IMG.shape[1]-IMG.shape[0],
                            IMG.shape[1]])*MIN), axis=0)

    if method=="area":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_AREA)
    if method=="linear":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_LINEAR)
    if method=="cubic":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_CUBIC)
    if method=="nearest":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_NEAREST)
    elif method=="lanc":
        IMG = cv2.resize(IMG, (height, width), interpolation=cv2.INTER_LANCZOS4)

    return(IMG)



################################################################################
################################################################################
def fix_ratio_to_csv(IMG, obj):
    Image_Dimension = IMG.shape
    if IMG.shape[0] > IMG.shape[1]:
        Image_needed_side_extention="V"
        Needed_addition = IMG.shape[0]-IMG.shape[1]
    else:
        Image_needed_side_extention="H"
        Needed_addition = IMG.shape[0]-IMG.shape[1]

    Data = pd.DataFrame({"Image_needed_side_extention":Image_needed_side_extention,
                "Needed_addition":Needed_addition, "Image_Dimension_X": Image_Dimension[0],
                "Image_Dimension_Y": Image_Dimension[1]}, index=[0])

    Path, File = os.path.split(obj.Case)
    if File[-4:] == ".dcm": File = File[:-4]
    saving_path = os.path.join(obj.output_path, File, "air_breast_mask")
    if not(os.path.isdir(saving_path)): os.makedirs(saving_path)
    Data.to_csv(os.path.join(saving_path, "fixing_ratio.csv"))



################################################################################
################################################################################
def bring_back_images_to_orginal_size(Path_to_csv, IMG, type="mask"):
    image_reset_info = pd.read_csv(Path_to_csv, sep=",", index_col=0)

    Image_needed_side_extention = image_reset_info["Image_needed_side_extention"].iloc[0]
    Needed_addition = image_reset_info["Needed_addition"].iloc[0]
    Image_Dimension_X = image_reset_info["Image_Dimension_X"].iloc[0]
    Image_Dimension_Y = image_reset_info["Image_Dimension_Y"].iloc[0]

    Max = max(Image_Dimension_X, Image_Dimension_Y)
    if type=="mask":
        IMG = cv2.resize((IMG*255).astype("uint8"), (Max, Max), interpolation = cv2.INTER_NEAREST)
    else:
        IMG = cv2.resize(IMG.astype("uint8"), (Max, Max), interpolation = cv2.INTER_NEAREST)

    if Image_needed_side_extention == "V":
        IMG = IMG[:, :Image_Dimension_Y]
    else:
        IMG = IMG[:Image_Dimension_X, :]

    if type=="mask":
        IMG[IMG>0] = 1
        IMG = (IMG*255).astype("uint8")

    return(IMG)



################################################################################
################################################################################
def bring_back_images_to_orginal_orientation(Path_to_csv, IMG):
    image_reset_orientation = pd.read_csv(Path_to_csv, sep=",", index_col=0)
    if ('FieldOfViewHorizontalFlip' in image_reset_orientation.columns) and not(
        pd.isnull(image_reset_orientation["FieldOfViewHorizontalFlip"].item())):
            if image_reset_orientation["ImageLaterality"].item() == 'L':
                IMG=np.fliplr(IMG)
            if image_reset_orientation["ImageLaterality"].item() == 'R':
                IMG=np.fliplr(IMG)

    return(IMG)



################################################################################
################################################################################
def Z_scoring(IMG, mask=[]):
    if len(mask)==0:
        mask = np.ones(IMG.shape)

    try:
        MEAN = IMG[mask].mean()
        STD = IMG[mask].std()

        IMG = (IMG-MEAN)/STD

    except:
        MEAN = IMG.mean()
        STD = IMG.std()

        IMG = (IMG-MEAN)/STD

    return(IMG)
