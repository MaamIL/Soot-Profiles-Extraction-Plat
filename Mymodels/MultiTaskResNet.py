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
class MultiTaskResNet(nn.Module):
    def __init__(self, config):
        super(MultiTaskResNet, self).__init__()
        
        
        self.config = config
        self.logger = CustomLogger(config.log_filename, self.__class__.__name__).get_logger()
        # self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # self.fc = nn.Linear(self.backbone.fc.in_features, self.config.in)  # or 204*804
        # Replace the final fully connected layer
        self.head_f = nn.Linear(self.backbone.fc.in_features, config.output_shape[0] * config.output_shape[1])  # For fv
        self.head_t = nn.Linear(self.backbone.fc.in_features, config.output_shape[0] * config.output_shape[1])  # For T
        
        # Set optimizer and scheduler here, after model is initialized
        config.optimizer = optim.Adam(self.backbone.fc.parameters(), lr=config.lr)
        config.scheduler = optim.lr_scheduler.StepLR(config.optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
        
        self.logger.info(f"""
                        \n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                          Model {config.model_name} initialized and moved to {config.device}
                          Backbone: {self.backbone}
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n
                          """)
        
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)

        out_f = self.head_f(features)
        out_t = self.head_t(features)
        out = torch.cat([out_f, out_t], dim=1)
        return out, None
    
    def plotLosses(self, train_losses, val_losses, test_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label=f'Training Loss ({train_losses[-1]:.3f})')
        plt.plot(val_losses, label=f'Validation Loss ({val_losses[-1]:.3f})')
        testx = self.config.num_epochs_stage1+self.config.num_epochs_stage2_step1+self.config.num_epochs_stage2_step2+self.config.num_epochs_stage2_step3
        plt.plot(testx,test_loss,  label=f"Test Loss ({test_loss:.3f})", marker='o')
        # Add labels only to the first and last points
        plt.text(0, train_losses[0], f"{train_losses[0]:.2f}", ha='center', va='bottom')
        # plt.text(len(train_losses)-1, train_losses[-1], f"{train_losses[-1]:.2f}", ha='center', va='bottom')
        
        plt.text(0, val_losses[0], f"{val_losses[0]:.2f}", ha='center', va='bottom')
        # plt.text(len(val_losses)-1, val_losses[-1], f"{val_losses[-1]:.2f}", ha='center', va='bottom')
        
        # plt.text(0, test_loss, f"{test_loss:.2f}", ha='center', va='bottom')
        #add vertical lines to indicate unfreezing stages
        # plt.axvline(x=self.config.num_epochs_stage1, color='r', linestyle='--', label='Stage 1 End. Unfreeze layer4')
        # plt.axvline(x=self.config.num_epochs_stage1+self.config.num_epochs_stage2_step1, color='g', linestyle='--', label='Unfreeze layer3')
        # plt.axvline(x=self.config.num_epochs_stage1+self.config.num_epochs_stage2_step1+self.config.num_epochs_stage2_step2, color='b', linestyle='--', label='Unfreeze layer2 and layer1')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config.out_dir, 'losses.png'))

        plt.ylim(0, 0.05)
        plt.legend()
        plt.savefig(os.path.join(self.config.out_dir, "losses_0_05.png"))
        plt.close()

    def train_model(self, train_loader, val_loader, test_loader):        
        self.train()
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        total_output_size = self.config.output_shape[0] * self.config.output_shape[1]

        for epoch in range(self.config.num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]") as pbar:
                for images, preds in pbar:
                    images = images.to(self.config.device)
                    preds = preds.to(self.config.device)
                    
                    # Zero the parameter gradients
                    self.config.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs, _ = self(images)
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
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Val]") as pbar:
                    for i, (images, preds) in enumerate(pbar):
                        images = images.to(self.config.device)
                        preds = preds.to(self.config.device)
                        
                        outputs , _= self(images)
                        # if "stage2_step3" in stage:
                        #     outputs[outputs < 0.1] = 0.0
                        # Compute loss
                        loss = self.config.criterion(outputs, preds)
                        
                        val_loss += loss.item()
                        pbar.set_postfix({"loss": loss.item()})
                        if epoch in (0,self.config.num_epochs-1,49,99,149):                     
                            #save heatmaps to epochs 0, 49, 99, 149
                            out_f = outputs[0][:total_output_size].view(self.config.output_shape)
                            out_t = outputs[0][total_output_size:].view(self.config.output_shape)
                            gt_f = preds[0][:total_output_size].view(self.config.output_shape)
                            gt_t = preds[0][total_output_size:].view(self.config.output_shape)
                            sample_dir = val_loader.dataset.dataset.sample_dirs[val_loader.sampler.data_source.indices[i]]
                            saveheatmaps(out_f, gt_f, epoch, str(i) + "_fv", images, self.config.out_dir, sample_dir)
                            saveheatmaps(out_t, gt_t, epoch, str(i) + "_T", images, self.config.out_dir, sample_dir)
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Step scheduler if provided
            if self.config.scheduler:
                self.config.scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), os.path.join(self.config.outdir,"best_flame_model.pth"))
                self.logger.info(f"Model saved with validation loss: {best_val_loss:.4f}")
            
            #test the model
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            with tqdm(test_loader, desc=f"Test Model [Test]") as pbar:
                for i, (inputs, gts) in enumerate(pbar):
                    inputs = inputs.to(self.config.device)
                    gts = gts.to(self.config.device)
                    
                    outputs, _ = self(inputs)
                    # outputs[outputs < ((1000 - 300) / (global_Fv_max - 300))] = 0.0  # Set values < 1000K to 0
                    loss = self.config.criterion(outputs, gts)
                    
                    test_loss += loss.item()
                    
                    # Save heatmaps for the first sample in the test set
                    saveheatmaps(outputs, gts, 0, "test"+str(i), inputs, self.config.out_dir, test_loader.dataset.dataset.sample_dirs[test_loader.sampler.data_source.indices[i]])
        test_loss /= len(test_loader)
        self.logger.info(f"Test Loss: {test_loss:.4f}")

        self.plotLosses(train_losses, val_losses, test_loss)
                        
        return train_losses, val_losses, test_loss, self

    