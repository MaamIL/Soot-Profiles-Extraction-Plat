import numpy as np
import os
import PIL.Image as Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from Logger import CustomLogger


# Custom Dataset: each subfolder in root_dir should contain CFDImage.mat and sootCalculation.mat
class FlameDataset(Dataset):

    def __init__(self, config):
        # custom_logger = CustomLogger(config)
        self.config = config
        self.logger = CustomLogger(config.log_filename, self.__class__.__name__).get_logger()
        self.logger.info("FlameDataset initialized with the provided configuration")
        self.sample_dirs = [os.path.join(config.root_dir, d) for d in os.listdir(config.root_dir)
                            if os.path.isdir(os.path.join(config.root_dir, d))] # List all subdirectories (each is one sample)
        self.logger.info(f"Found {len(self.sample_dirs)} samples in the dataset.")
    def __len__(self):
        return len(self.sample_dirs)
    
    # def compute_mean_std(self):
    #     """Compute mean and standard deviation of the dataset images."""
    #     mean = 0.0
    #     std = 0.0
    #     for idx in range(len(self)):
    #         image, _ = self[idx]  # Get the image from __getitem__
    #         mean += image.mean([1, 2])  # Compute mean across height and width for each channel
    #         std += image.std([1, 2])    # Compute std across height and width for each channel

    #     mean /= len(self)
    #     std /= len(self)
    #     self.logger.info(f"Computed Mean: {mean}, Std: {std}")
    #     return mean, std
    
    def _getFv_(self, soot_mat):
        """Extract fv from sootCalculation.mat."""
        fv = soot_mat["fv"]
        # Set values smaller than 0.1 to 0
        fv[fv < self.config.setFvValZero] = 0.0
        #Normalize fv
        if(self.config.isNorm):
            fv = (fv - self.config.global_fv_min)/max((self.config.global_fv_max-self.config.global_fv_min), 1e-6)
        # Pad fv to the desired shape (202, 92)
        fv = np.pad(fv, ((0, self.config.output_shape[0] - fv.shape[0]), (0, self.config.output_shape[1] - fv.shape[1])), mode='constant', constant_values=0)
        # if(self.config.isNorm):
        #     # Normalize fv
        #     if self.config.global_fv_max == 0:
        #         self.logger.error(f"Global max value for fv not set properly. Current value: {self.config.global_fv_max}")
        #         exception = ValueError("Global max value for fv not set properly.")
        #         raise exception          
        #     fv = fv / self.config.global_fv_max        
        return fv
    
    def _getT_(self, soot_mat):
        """Extract fv from sootCalculation.mat."""
        T = soot_mat["T"]
        T[T < self.config.setTValZero] = 300.0    
        #Normalize T
        if(self.config.isNorm):
            T = (T - self.config.global_T_min)/max((self.config.global_T_max-self.config.global_T_min), 1e-6)
            
        #Temperature padding needs to be with values of 300 (kalvin as the minimum value is 300)- put 0.0 because it is normalized
        T  = np.pad(T,((0, self.config.output_shape[0] - T.shape[0]), (0, self.config.output_shape[1] - T.shape[1])), mode='constant', constant_values=0.0)   # shape: (202, 92)
        
            # # Normalize
            # if self.config.global_T_max == 300:# or self.global_fv_max == 0:
            #     self.logger.error(f"Global max values not set properly. T_max: {self.config.global_T_max}")
            #     exception = ValueError("Global max value for T not set properly.")
            #     raise exception
            # T = (T - 300) / (self.config.global_T_max - 300)      
        return T
    
    def __getitem__(self, idx):        
        sample_dir = self.sample_dirs[idx]
        
        cfd_path = os.path.join(sample_dir, "CFDImage.mat")
        cfd_mat = sio.loadmat(cfd_path)
        image_array = cfd_mat["CFDImage"].astype(np.float32)
        image_array = np.flipud(image_array)  # Flip the image array vertically
        image_array[image_array < self.config.setImgValZero] = 0.0 #negative values are not relevant and are set to 0.0
        #normelize
        # image_array = image_array/4095.0
        image_array = (image_array-self.config.global_img_min)/max((self.config.global_img_max-self.config.global_img_min), 1e-6)  # Avoid division by zero
        # image_array = (image_array)/max((self.config.global_img_max), 1e-6)  # Avoid division by zero
        #padding
        image = np.pad(image_array,((0,self.config.input_shape[1]-image_array.shape[0]),(0,self.config.input_shape[2]-image_array.shape[1]),(0,0)), mode='constant', constant_values=0)

        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.convert("RGB")                
        image = transforms.ToTensor()(image)

        soot_path = os.path.join(sample_dir, "sootCalculation.mat")
        soot_mat = sio.loadmat(soot_path)
        if self.config.targetType == "T":
            T = self._getT_(soot_mat)
            target = torch.tensor(T, dtype=torch.float32)
        elif self.config.targetType == "fv":
            fv = self._getFv_(soot_mat)
            target = torch.tensor(fv, dtype=torch.float32)
        elif self.config.targetType == "both":
            fv = self._getFv_(soot_mat)
            T = self._getT_(soot_mat)
            target = np.stack([fv, T], axis=0)  # shape: (2, 202, 92)
            # target = target.flatten()            # shape: (2*202*92,) i.e., (35696,)
            target = torch.tensor(target, dtype=torch.float32)        
        
        return image, target
       
        
