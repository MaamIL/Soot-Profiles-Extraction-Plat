import torch
import torch.nn as nn
import torch.optim as optim
from Logger import CustomLogger
from Plot_Outputs import saveheatmaps
from tqdm import tqdm
from torchvision import models
import os
import matplotlib.pyplot as plt


# Combine backbone and heads into a multi-task network
class TwoStageTraining(nn.Module):
    def __init__(self, config):
        super(TwoStageTraining, self).__init__()
        self.logger = CustomLogger(config.log_filename, self.__class__.__name__).get_logger()
        self.config = config
        self.num_epochs_stage1 = 5 # Number of epochs for stage 1
        self.num_epochs_stage2_step1 = 5 # Number of epochs for stage 2 step 1 (unfreeze layer4)
        self.num_epochs_stage2_step2 = 5 # Number of epochs for stage 2 step 2 (unfreeze layer3)
        self.num_epochs_stage2_step3 = 5 # Number of epochs for stage 2 step 3 (unfreeze layer2 and layer1)
        # Load pre-trained ResNet18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace final layer with custom decoder
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.logger.info(f"""
                        \n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                          Model {config.model_name} initialized and moved to {config.device}
                          Backbone: {self.backbone}
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n
                          """)
        
        # Decoder to map features to temperature map
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 1024),
            # nn.ReLU(),  #TODO: TRY SIGMOID
            nn.Sigmoid(),
            nn.Linear(1024, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, self.config.output_shape[0] * self.config.output_shape[1])
        )
        
        # Initialize weights for decoder
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Freeze backbone initially
        self.freeze_backbone()
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_layer(self, layer_name):
        if layer_name == "layer4":
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        elif layer_name == "layer3":
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
        elif layer_name == "layer2":
            for param in self.backbone.layer2.parameters():
                param.requires_grad = True
        elif layer_name == "layer1":
            for param in self.backbone.layer1.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.decoder(features)
        return output.view(-1, self.config.output_shape[0], self.config.output_shape[1])
    
    def plotLosses(self, train_losses, val_losses, test_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label=f'Training Loss ({train_losses[-1]:.3f})')
        plt.plot(val_losses, label=f'Validation Loss ({val_losses[-1]:.3f})')
        testx = self.num_epochs_stage1+self.num_epochs_stage2_step1+self.num_epochs_stage2_step2+self.num_epochs_stage2_step3
        plt.plot(testx,test_loss,  label=f"Test Loss ({test_loss:.3f})", marker='o')
        # Add labels only to the first and last points
        plt.text(0, train_losses[0], f"{train_losses[0]:.2f}", ha='center', va='bottom')
        # plt.text(len(train_losses)-1, train_losses[-1], f"{train_losses[-1]:.2f}", ha='center', va='bottom')
        
        plt.text(0, val_losses[0], f"{val_losses[0]:.2f}", ha='center', va='bottom')
        # plt.text(len(val_losses)-1, val_losses[-1], f"{val_losses[-1]:.2f}", ha='center', va='bottom')
        
        # plt.text(0, test_loss, f"{test_loss:.2f}", ha='center', va='bottom')
        #add vertical lines to indicate unfreezing stages
        plt.axvline(x=self.num_epochs_stage1, color='r', linestyle='--', label='Stage 1 End. Unfreeze layer4')
        plt.axvline(x=self.num_epochs_stage1+self.num_epochs_stage2_step1, color='g', linestyle='--', label='Unfreeze layer3')
        plt.axvline(x=self.num_epochs_stage1+self.num_epochs_stage2_step1+self.num_epochs_stage2_step2, color='b', linestyle='--', label='Unfreeze layer2 and layer1')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config.out_dir, 'losses.png'))

        plt.ylim(0, 0.05)
        plt.legend()
        plt.savefig(os.path.join(self.config.out_dir, "losses_0_05.png"))
        plt.close()

    # Training function
    def train_model_func(self, stage, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
                for images, preds in pbar:
                    images = images.to(self.config.device)
                    preds = preds.to(self.config.device)
                    
                    # Zero the parameter gradients
                    self.config.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self(images)
                    # if "stage2_step3" in stage:
                    #     outputs[outputs < 0.1] = 0.0  # Set values < 1000K to 0
                    # Compute loss
                    loss = self.config.criterion(outputs, preds)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.config.optimizer.step()
                    
                    running_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
            
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                    for i, (images, preds) in enumerate(pbar):
                        images = images.to(self.config.device)
                        preds = preds.to(self.config.device)
                        
                        outputs = self(images)
                        # if "stage2_step3" in stage:
                        #     outputs[outputs < 0.1] = 0.0
                        # Compute loss
                        loss = self.config.criterion(outputs, preds)
                        
                        val_loss += loss.item()
                        pbar.set_postfix({"loss": loss.item()})
                        if epoch in (0,num_epochs-1,49,99,149):                     
                            #save heatmaps to epochs 0, 49, 99, 149
                            saveheatmaps(outputs, preds, epoch, str(i)+"_"+stage, images, self.config.out_dir, val_loader.dataset.dataset.sample_dirs[val_loader.sampler.data_source.indices[i]], self.config)
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Step scheduler if provided
            if self.config.scheduler:
                self.config.scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), os.path.join(self.config.out_dir,"best_flame_model.pth"))
                self.logger.info(f"Model saved with validation loss: {best_val_loss:.4f}")
                best_model = self
        
        return train_losses, val_losses, best_model
        
    def train_model(self, train_loader, val_loader, test_loader):
        
        # Stage 1: Train only the decoder with frozen backbone
        self.logger.info("Stage 1: Training decoder only...")
        self.logger.info(f"lr={self.config.lr}")
        self.config.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr)
        self.config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, mode='min', factor=0.5, patience=2)
        
        train_losses1, val_losses1, _ = self.train_model_func("stage1", train_loader, val_loader, self.num_epochs_stage1)
        
        # Stage 2: Gradual unfreezing of backbone layers
        self.logger.info("Stage 2: Gradual unfreezing...")
        
        # Step 1: Unfreeze layer4
        self.logger.info("Unfreezing layer4...")
        self.logger.info(f"lr={self.config.lr/ 20}")
        self.unfreeze_layer("layer4")
        self.config.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr / 20)#10)
        self.config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, mode='min', factor=0.5, patience=1)
        
        train_losses2_1, val_losses2_1, _ = self.train_model_func("stage2_step1", train_loader, val_loader, self.num_epochs_stage2_step1)
        
        # Step 2: Unfreeze layer3
        self.logger.info("Unfreezing layer3...")
        self.logger.info(f"lr={self.config.lr/ 50}")
        self.unfreeze_layer("layer3")
        self.config.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr / 50)#20)
        self.config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, mode='min', factor=0.5, patience=1)

        train_losses2_2, val_losses2_2, _ = self.train_model_func("stage2_step2", train_loader, val_loader, self.num_epochs_stage2_step2)
        
        # Step 3: Unfreeze layer2 and layer1
        self.logger.info("Unfreezing remaining layers...")
        self.logger.info(f"lr={self.config.lr/ 100}")
        self.unfreeze_layer("layer2")
        self.unfreeze_layer("layer1")
        self.config.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config.lr/ 100) #50)
        self.config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, mode='min', factor=0.5, patience=1)
        
        train_losses2_3, val_losses2_3, best_model = self.train_model_func("stage2_step3", train_loader, val_loader, self.num_epochs_stage2_step3)
        
        # Combine all losses for plotting
        train_losses = train_losses1 + train_losses2_1 + train_losses2_2 + train_losses2_3
        val_losses = val_losses1 + val_losses2_1 + val_losses2_2 + val_losses2_3
        
        #test the model
        self.eval()
        test_loss = 0.0

        with torch.no_grad():
            with tqdm(test_loader, desc=f"Test Model [Test]") as pbar:
                for i, (inputs, gts) in enumerate(pbar):
                    inputs = inputs.to(self.config.device)
                    gts = gts.to(self.config.device)
                    
                    outputs = best_model(inputs)
                    # outputs[outputs < ((1000 - 300) / (global_Fv_max - 300))] = 0.0  # Set values < 1000K to 0
                    loss = self.config.criterion(outputs, gts)
                    
                    test_loss += loss.item()
                    
                    # Save heatmaps for the first sample in the test set
                    saveheatmaps(outputs, gts, 0, "TEST"+str(i), inputs, self.config.out_dir, test_loader.dataset.dataset.sample_dirs[test_loader.sampler.data_source.indices[i]],self.config)
        test_loss /= len(test_loader)
        self.logger.info(f"Test Loss: {test_loss:.4f}")

        return train_losses, val_losses, test_loss, best_model