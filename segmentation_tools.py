import pdb
import numpy as np
from scipy.ndimage import label
from skimage.util import img_as_float
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


# helpful slic tool
def FSLIC(IMAGE, IM, NumSLIC, ComSLIC, SigSLIC, Initial=False):
    IMAGE = img_as_float(IMAGE)
    Segments = slic(IMAGE, n_segments=NumSLIC, sigma=SigSLIC, compactness=ComSLIC)
    if Initial == True:  # if initial is true, it returns the fusedimage showing the segments
        Fusied_Image = mark_boundaries(IM, Segments, color = (1, 1, 1))
        return (Segments, Fusied_Image[...,0])
    else:
        return (Segments)



# normalize image between zero and range provided
def Normalize_Image(IMAGE, Range, Min=None, Max=None, flag_max_edition=None, flag_min_edition=None, bits_conversion=None):
    IMAGE = IMAGE.astype('float')

    if Min==None: Min = IMAGE.min()
    if Max==None: Max = IMAGE.max()

    if Min != Max:
        Out_Img = (IMAGE-Min)/(Max-Min)
    else:
        Out_Img = np.ones(Out_Img.shape)

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

    return(Out_Img)



# trimming the image and its mask if it is given
def cutting_image(IMG, skipping_rows, MASK=[], x_ratio=1, y_ratio=1, skip_columns_to=[]):
    if len(skip_columns_to) == 0:
        if round(IMG.shape[0]*x_ratio) * round(IMG.shape[1]*y_ratio) %2 == 0:
            IMG = IMG[skipping_rows:round(IMG.shape[0]*x_ratio),
                      :round(IMG.shape[1]*y_ratio)]

            if len(MASK)>0:
                MASK = MASK[skipping_rows:round(MASK.shape[0]*x_ratio),
                            :round(MASK.shape[1]*y_ratio)]
        else:
            IMG = IMG[skipping_rows:round(IMG.shape[0]*x_ratio),
                      :round(IMG.shape[1]*y_ratio)-1]

            if len(MASK)>0:
                MASK = MASK[skipping_rows:round(MASK.shape[0]*x_ratio),
                            :round(MASK.shape[1]*y_ratio)-1]

    else:
        IMG = IMG[skipping_rows:round(IMG.shape[0]*x_ratio),
                  :int(skip_columns_to[0])]

        if len(MASK)>0:
            MASK = MASK[skipping_rows:round(MASK.shape[0]*x_ratio),
                        :int(skip_columns_to[0])]

    return(IMG, MASK)



# in a mask find the largest object
def find_largest_obj(Mask):
    # one shows background and zero objects
    temp_mask = (np.logical_not(Mask)).astype("int")
    # make the first and last row zero to make sure it is not affected by noise
    temp_mask[0,:] = 0
    labeled_obj = label(temp_mask)[0]

    if labeled_obj.max()>1:
        BG_Label = labeled_obj[0, -1]
        Unique_labels, counts = np.unique(labeled_obj, return_counts=True)
        counts = np.delete(counts, np.where(Unique_labels==BG_Label), None)
        Unique_labels = np.delete(Unique_labels, np.where(Unique_labels==BG_Label), None)
        Max = Unique_labels[counts.argmax()]
        Mask[labeled_obj!=Max] = True
    return(Mask)
