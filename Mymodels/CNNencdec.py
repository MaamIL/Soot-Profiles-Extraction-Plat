
from Logger import CustomLogger
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from Plot_Outputs import saveheatmaps, save_error_heatmaps
import torch.optim as optim
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNNencdec(nn.Module):
    def __init__(self, config):
        super(CNNencdec, self).__init__()
        self.config = config        
        self.logger = CustomLogger(self.config.log_filename, self.__class__.__name__).get_logger()
        
        in_channels = self.config.input_shape[0]
        out_channels = 2 if self.config.targetType == "both" else 1

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        # self.config.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr), weight_decay=1e-4) #added weight decay (L2 regularization) to optimizer
        self.config.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        # scheduler
        self.config.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.config.optimizer, mode='min', factor=0.3, patience=3) #reduce lr in case of plateau in validation loss (no improvements after 5 epochs), then learning rate will be reduced: new_lr = lr * factor

        # self.config.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.config.optimizer,
        #     step_size=self.config.scheduler_step,
        #     gamma=self.config.scheduler_gamma
        # )

        # # Show model summary
        # self.show_model_summary()

        self.logger.info(f"""
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Model '{config.model_name}' initialized on {config.device}
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """)

    def forward(self, x):
        enc_out, features = self.encoder(x)
        dec_out = self.decoder(enc_out, features)
        out = self.final(dec_out)
        out = torch.sigmoid(out)
        # Match output to desired size (e.g. 727, 138)
        target_h, target_w = self.config.output_shape
        if out.shape[2] != target_h or out.shape[3] != target_w:
            out = F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return out

    def train_model(self, train_loader, val_loader, test_loader):
        self.to(self.config.device)
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(self.config.num_epochs):
            self.train()
            running_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]") as pbar:
                for inputs, gts in pbar:
                    inputs = inputs.to(self.config.device)
                    gts = gts.to(self.config.device)

                    self.config.optimizer.zero_grad()
                    outputs = self(inputs)  
                    # loss = self.config.criterion(outputs, preds)
                    if self.config.targetType == "both":
                        loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
                        loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
                        loss = 0.5 * loss_fv + 0.5 * loss_T
                    elif self.config.targetType == "fv":
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    elif self.config.targetType == "T":
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    
                    loss.backward()
                    self.config.optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})                    

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.eval()
            val_loss = 0.0
            print4samples = 0
            early_stop_patience = 15 # Number of epochs to wait before early stopping
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Val]") as pbar:
                    for i, (inputs, gts) in enumerate(pbar):
                        inputs = inputs.to(self.config.device)
                        gts = gts.to(self.config.device)

                        outputs = self(inputs)  
                        # loss = self.config.criterion(outputs, preds)
                        if self.config.targetType == "both":
                            loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
                            loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
                            loss = 0.5 * loss_fv + 0.5 * loss_T
                        elif self.config.targetType == "fv":
                            loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                        elif self.config.targetType == "T":
                            loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                        val_loss += loss.item()
                        pbar.set_postfix({"loss": loss.item()})
                       

                        if (epoch%10 == 0) and print4samples < 4:#(epoch in (0, int(self.config.num_epochs/4), int(self.config.num_epochs/2), int(self.config.num_epochs*0.75), self.config.num_epochs - 1))) and print4samples < 4:
                            saveheatmaps(outputs, gts, epoch, str(i)+"_", inputs,
                                         self.config.out_dir,
                                         val_loader.dataset.dataset.sample_dirs[val_loader.sampler.data_source.indices[i]],
                                         self.config)      
                            save_error_heatmaps(outputs, gts, epoch, str(i)+"_", inputs, self.config.out_dir,
                                val_loader.dataset.dataset.sample_dirs[val_loader.sampler.data_source.indices[i]],
                                self.config, loss.item())
                            print4samples += 1
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            self.logger.info(f"Epoch {epoch+1}, lr: {self.config.optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f} (best: {best_val_loss:.8f})")

            if hasattr(self.config, 'scheduler') and self.config.scheduler:
                self.config.scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), os.path.join(self.config.out_dir, "best_flame_model.pth"))
                self.logger.info(f"Best model saved with val loss: {best_val_loss:.8f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1} due to no improvement in val loss for {early_stop_patience} epochs.")
                break
            torch.cuda.empty_cache()

        # Test the model
        self.logger.info(f"\n\nTesting model on Best model saved with val loss: {best_val_loss:.8f}")
        self.load_state_dict(torch.load(os.path.join(self.config.out_dir, "best_flame_model.pth")))
        self.to(self.config.device)
        self.eval()
        test_loss = 0.0
        # test_lossb40 = 0.0
        print5samples = 0
        with torch.no_grad():
            with tqdm(test_loader, desc="Testing") as pbar:
                for i, (inputs, gts) in enumerate(pbar):
                    inputs = inputs.to(self.config.device)
                    gts = gts.to(self.config.device)
                    outputs = self(inputs)  
                    
                    # loss = self.config.criterion(outputs, gts)
                    # Calculate losses for both fv and T 
                    if self.config.targetType == "both":
                        normalized_setFvValZero = (self.config.setFvValZero - self.config.global_fv_min) / max((self.config.global_fv_max - self.config.global_fv_min), 1e-6)
                        normalized_setTValZero = (self.config.setTValZero - self.config.global_T_min) / max((self.config.global_T_max - self.config.global_T_min), 1e-6)
                        outputs[:, 0, :, :][outputs[:, 0, :, :] < normalized_setFvValZero] = 0.0
                        outputs[:, 1, :, :][outputs[:, 1, :, :] < normalized_setTValZero] = 0.0
                        loss_fv = F.mse_loss(outputs[:, 0, :, :], gts[:, 0, :, :])
                        loss_T = F.mse_loss(outputs[:, 1, :, :], gts[:, 1, :, :])
                        loss = 0.5 * loss_fv + 0.5 * loss_T
                    elif self.config.targetType == "fv":
                        loss_b40 = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                        normalized_setFvValZero = (self.config.setFvValZero - self.config.global_fv_min) / max((self.config.global_fv_max - self.config.global_fv_min), 1e-6)
                        outputs[:, 0, :, :][outputs[:, 0, :, :] < normalized_setFvValZero] = 0.0
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    elif self.config.targetType == "T":
                        loss_b40 = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                        normalized_setTValZero = (self.config.setTValZero - self.config.global_T_min) / max((self.config.global_T_max - self.config.global_T_min), 1e-6)
                        outputs[:, 0, :, :][outputs[:, 0, :, :] < normalized_setTValZero] = 0.0
                        loss = F.mse_loss(outputs[:, 0, :, :], gts[:, :, :])
                    test_loss += loss.item()
                    # test_lossb40 += loss_b40.item() 
                    if print5samples < 10:
                        saveheatmaps(outputs, gts, 0, "test"+str(i), inputs,
                                    self.config.out_dir,
                                    test_loader.dataset.dataset.sample_dirs[test_loader.sampler.data_source.indices[i]],
                                    self.config)
                        save_error_heatmaps(outputs, gts, 0, "test"+str(i)+"_", inputs, self.config.out_dir,
                            test_loader.dataset.dataset.sample_dirs[test_loader.sampler.data_source.indices[i]],
                            self.config, loss.item())
                        print5samples += 1
        test_loss /= len(test_loader)
        # test_lossb40 /= len(test_loader) if self.config.targetType == "fv" else 1.0
        # self.logger.info(f"Test Loss before 0: {test_lossb40:.8f}")
        self.logger.info(f"Test Loss: {test_loss:.8f}")

        self.plotLosses(train_losses, val_losses, test_loss)
        return train_losses, val_losses, test_loss, self

    def plotLosses(self, train_losses, val_losses, test_loss):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.axhline(test_loss, color='red', linestyle='--', label=f'Test Loss: {test_loss:.8f}')
        plt.title("Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.out_dir, "losses.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.axhline(test_loss, color='red', linestyle='--', label=f'Test Loss: {test_loss:.8f}')
        plt.title("Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.ylim(0, 0.0001)  # Zoom in on the y-axis
        plt.savefig(os.path.join(self.config.out_dir, "losses_zoom.png"))
        plt.close()
        
    def show_model_summary(self):
        try:
            dummy_input = torch.randn(1, *self.config.input_shape).to(self.config.device)
            output = self(dummy_input)  # forward pass
            
        except Exception as e:
            self.logger.error(f"Model summary failed: {e}")


### NETWORK MODULES ###

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return self.relu(out)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=2),
            # ResidualBlock(512, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )

    def forward(self, x):
        x0 = self.initial(x)
        features = [x0]
        for block in self.blocks:
            x0 = block(x0)
            features.append(x0)
        return x0, features[::-1]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = nn.ModuleList([
            # self._up_block(512, 512, 0.1),
            self._up_block(512, 512, 0.2),
            self._up_block(512, 512, 0.3),
            self._up_block(512, 256, 0.3),
            self._up_block(256, 128, 0.2),
            self._up_block(128, 64, 0.1)
        ])

    def _up_block(self, in_ch, out_ch, dropout=0.2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout)  # Optional dropout for regularization
        )

    def forward(self, x, features):
        for i, up in enumerate(self.up_blocks):
            x = up(x)
            if i + 1 < len(features):
                enc_feat = features[i + 1]
                if x.shape != enc_feat.shape:
                    enc_feat = F.interpolate(enc_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + enc_feat  # Skip connection
        return x
