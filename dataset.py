# ### Dataset
# - Custom Dataset
# - Task:
#     1. Create a list of all 2D slices
#     2. Extract & load slice and corresponding label mask
#     3. Data Augmentation. Augment slice and mask identically
#     4. Return augmented slice and mask
# - Data Augmentation:
#     - Scaling (0.85, 1.15)
#     - Rotation (-45°, 45)
#     - ElasticTransformation: moving the pixels locally arround using a displacement field

from pathlib import Path
import torch
import numpy as np
import imgaug # random seed
from imgaug.augmentables.segmaps import SegmentationMapsOnImage # Augment segmentation and slice in same way

"""
1- create a list containing path to all slices, will use glob fun. 2- extract corresponding label path for each slice
3- use paths to load slices and corresponding segmentation given an index
4- Apply Data Augmentation on segment and mask Identically . Then return the slice and mask
"""

class CardiacDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, augment_params):
        self.all_files = self.extract_files(root)  # store path to all slices 
        self.augment_params = augment_params # assigning augment_params to the corresponding class attribute
        
    @staticmethod             # Never accesses any class attribute, No need for self argument
    def extract_files(root):  
        files= []                       # list contain path to all 2D slices
        for subject in root.glob("*"):  # path to all subjects
            slice_path = subject/"data"  # full path to numpy files of this subject
            for slice in slice_path.glob("*.npy"):  # only npy files
                files.append(slice)
        return files
    
    @staticmethod
    def change_img_to_label_path(path): # replaces imagesTr(slice path) with labelsTr(segmentation path)
                                        # extract label path for each slice
                                    
        parts = list(path.parts) # get all directories within the path
        parts[parts.index("data")] = "masks" # Replace data(preprocessed slice) with masks
        return Path(*parts) # Combine list back into a Path object
    
    def augment(self, slice, mask):
        random_seed = torch.randint(0, 100000, (1, )).item() # get the item(scaler) from this tensor with shape (1, )
        imgaug.seed(random_seed)  # random augmentations besides working with pytorch data loader
        
        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.augment_params(image=slice, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()  # actual array from the augmented mask
        return slice_aug, mask_aug
    
    def __len__(self):
        return len(self.all_files)   # length of all files list
    
    def __getitem__(self, idx):   # the item with this idx
        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        slice = np.load(file_path).astype(np.float32)
        mask = np.load(mask_path)
        
        if self.augment_params:          # if augmentation pipeline exists, 
            slice, mask = self.augment(slice, mask)  # augment slice and mask
        return np.expand_dims(slice, 0), np.expand_dims(mask, 0)   # return the data
    
       # np.expand_dims(slice, 0) > array(slice) and dim-channel we want to add(0), dont have to call unsqueeze all the time