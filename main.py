#This file holds the configurations regarding the data creation and the model selection.
#It also holds the main function that runs the training or inference of the model.

import os
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from importlib import import_module

from Logger import CustomLogger
from DataCreation import FlameDataset
from Plot_Outputs import saveheatmaps
from Powerpoint_output_visuals import PowerPointVisual


# Configuration
class Config:
    
    def __init__(self):
        #Params for dataset creation    
        self.paramsType2 = "~~~~~Params for dataset creation~~~~~"
        self.root_dir = 'C:/Users/User/Documents/GenerateData/GeneratedData'  # Path to your dataset
        self.global_T_max = 2853.0
        self.global_fv_max = 11.224797513519933
        self.Fvmax_height = 808
        self.Fvmax_width = 213
        self.Imagemax_height = 808
        self.Imagemax_width = 213
        # self.global_T_max = max([
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["T"].max() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #max temp in the dataset for normalization
        # self.global_fv_max = max([
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["fv"].max() for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # ]) #max Fv in the dataset for normalization  
        # #find max Fv size in dir
        # self.Fvmax_height = max(
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["fv"].shape[0]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # )
        # self.Fvmax_width = max(
        #     sio.loadmat(os.path.join(self.root_dir, d, "sootCalculation.mat"))["fv"].shape[1]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))
        # )
        # self.Imagemax_height = max(
        #     (sio.loadmat(os.path.join(self.root_dir, d, "CFDImage.mat"))["CFDImage"].shape[0]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)))
        # )
        # self.Imagemax_width = max(
        #     (sio.loadmat(os.path.join(self.root_dir, d, "CFDImage.mat"))["CFDImage"].shape[1]
        #     for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)))
        # )       
        self.input_shape = (3, self.Imagemax_height, self.Imagemax_width)  # (C, H, W) for input RGB flame images
        self.output_shape = (self.Fvmax_height, self.Fvmax_width) # (Height, Width) for temperature maps
        self.targetType = "both" # "T", "fv", or "both"
        self.isNorm = True # True/False - Normalize the input images
        self.setImgValZero = 0#50 #Set values smaller than 50 to 0 in CFDImage
        self.setFvValZero = 0.01 #Set values smaller than 0.01 to 0 in sootCalculation["fv"]
        self.setTValZero = 1000.0 #Set values smaller than 1000 to 300.0 in sootCalculation["T"]
    # Params for model training
        self.paramsType3 = "~~~~~Params for model training~~~~~"
        self.model_name = "CNNencdec" #"TwoStageTraining" / "MultiTaskResNet" / "CNNencdec" 
        self.batch_size = 12 # Batch size for training
        self.criterion = nn.MSELoss()
        self.lr=0.0001
        self.num_epochs = 350 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = ""
        self.scheduler_step = 10
        self.scheduler_gamma = 0.02
    #Params for outputs and logging
        self.paramsType1 = "~~~~~Params for outputs and logging~~~~~"     
        self.out_dir = f'Outputs_{self.model_name}_{self.targetType}_{time.strftime("%Y-%m%d-%H%M%S")}' #Path to save outputs   
        os.makedirs(self.out_dir, exist_ok=True) # Create output directory    
        self.log_filename = os.path.join(self.out_dir, "log.txt")
        self.logger = CustomLogger(self.log_filename, self.__class__.__name__).get_logger()
        self.savePlots = True #True/False - Save plots of the training process or show them without saving


    def print_config(self):
        """Print the configuration settings."""
        self.logger.info(f"""Logging to {self.log_filename}
                                          Log Format: {config.logger.logger.handlers[1].formatter._fmt}""")
        
        txt = f"\n\n~~~~~~~~~~~~~~Configuration settings~~~~~~~~~~~~~~~~~\n"
        for attr, value in self.__dict__.items():
            txt += f"{attr} = {value}\n"
        txt += f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        self.logger.info(txt)   

############################################################Functions

# Function to prepare data
def prepare_data(config):
    """
    Prepare and split the data for training
    
    Args:
        flame_images: List of RGB flame images (1857, 813, 3)
        temperature_maps: List of temperature maps (202, 92)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # # Define transformations
    # transform = transforms.Compose([
    #     transforms.RandomApply([
    #         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)
    #     ], p=0.8),
        
    #     transforms.RandomAffine(
    #         degrees=0,               # No rotation
    #         translate=(0, 0.05),     # Vertical shift only
    #         scale=(0.9, 1.1),        # Slight zoom in/out
    #         shear=0                  # No shear
    #     ),
        
    #     transforms.RandomResizedCrop(
    #         size=(224, 224),         # Or your desired size
    #         scale=(0.85, 1.0),       # Don't crop too aggressively
    #         ratio=(0.9, 1.1)         # Keep it nearly square (or shape of your input)
    #     ),

    #     transforms.ToTensor(),

    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],  # ImageNet stats (for pretrained models)
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    
    dataset = FlameDataset(config)
    
    train_size = int(0.7 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size
    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    config.logger.info(f"~~~~~~~~~~~~\nDataset sizes: \nTrain: {len(train_dataset)} \nValidation: {len(val_dataset)} \nTest: {len(test_dataset)}\n~~~~~~~~~~~~~~~~")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    return train_loader, val_loader, test_loader


############################################################
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    #config
    config = Config()
    config.print_config()
    # Create a logger for main
    main_logger = CustomLogger(config.log_filename, __name__).get_logger()

  #1. Create Data Loaders
    main_logger.info("Creating dataset...")
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)

 #2. Run Model    
    
    main_logger.info("Running model...")

    try:
        model_module = import_module(f"Mymodels.{config.model_name}")
        model_class = getattr(model_module, config.model_name)
        model = model_class(config).to(config.device)

        train_losses, val_losses, test_loss, best_model = model.train_model(train_loader, val_loader, test_loader)
        main_logger.info(f"Model '{config.model_name}' trained successfully.")
        main_logger.info(f"Best model parameters:\n~~~~~~~~~~~~~~~\n{best_model.parameters()}\n saved to {os.path.join(config.out_dir, 'best_model.pth')}")
        
    except Exception as e:
        main_logger.error(f"Error loading model '{config.model_name}': {e}")
        raise

  #3. Save outputs
    try:
        model.plotLosses(train_losses, val_losses, test_loss)
        main_logger.info(f"Loss plots saved to {os.path.join(config.out_dir, 'losses.png')}")
    except Exception as e:
        main_logger.error(f"Error in saving loss plots: {e}")
        
    # # Create epoch visualization PowerPoint presentation
    # try:        
    #     ppt_vis = PowerPointVisual(config, train_losses, val_losses, test_loss)
    #     epoch_images = ppt_vis.collect_images()
    #     ppt_vis.create_presentation(epoch_images, 'losses.png', output_file="EpochsPresentation.pptx")
    #     main_logger.info(f"Presentation saved to {os.path.join(config.out_dir, 'EpochsPresentation.pptx')}")
    # except Exception as e:
    #     main_logger.error(f"Error in creating presentation: {e}")

    
    

    



