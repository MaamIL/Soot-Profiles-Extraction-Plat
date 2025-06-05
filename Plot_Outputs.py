import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def heatmaps(r, z, fvORt, cbar_label, pltTitle, savefile):
    # Ensure r and z are 1D arrays matching the dimensions of fvORt
    r = np.linspace(0, 1, fvORt.shape[1]) if np.isscalar(r) or len(r) != fvORt.shape[1] else r
    z = np.linspace(0, 1, fvORt.shape[0]) if np.isscalar(z) or len(z) != fvORt.shape[0] else z
    R, Z = np.meshgrid(r, z)
    plt.figure(figsize=(16, 12))
    levels = np.linspace(0, 1, 51)  # 50 intervals between 0 and 1
    contour = plt.contourf(R, Z, fvORt, levels=levels, cmap='inferno', vmin=0, vmax=1)
    plt.gca().set_aspect('equal')
    cbar = plt.colorbar(contour)
    cbar.set_label(cbar_label)
    plt.xlabel("Radial Position (r) [mm]")
    plt.ylabel("Axial Position (z) [mm]")
    plt.title(pltTitle)
    plt.savefig(savefile)
    plt.close()

def saveheatmaps(outputs, gts, epoch, sample_number, inputs, heat_dir, sample_dir, config):
    samp_folder = sample_dir[sample_dir.rfind('\\')+1:]

    # Convert tensors to CPU and detach for processing
    preds = outputs[0].cpu().detach()
    gts = gts[0].cpu().detach()
    inputs = inputs[0].cpu().detach()
    from PIL import Image
    # Optional image save (only once, epoch=0 or test sample)
    if epoch == 0 or ("test" in sample_number):
        # Save input image
        # to_pil = transforms.ToPILImage()
        if inputs.ndimension() == 3 and inputs.shape[0] == 3:
            # image_array = inputs.cpu().detach().numpy().astype(np.float32)
            # image_array = (image_array/np.max(image_array))
            
            # image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("RGB")
            # # image = to_pil(inputs).convert("RGB")
            # image.save(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Input.jpg'))
            # Convert from PyTorch tensor (C, H, W) to NumPy array (H, W, C)
            image_array = inputs.cpu().detach().numpy().astype(np.float32)
            image_array = np.transpose(image_array, (1, 2, 0))  # (H, W, C)

            # Normalize to [0, 1] and scale to [0, 255]
            image_array = image_array / np.max(image_array)
            image_uint8 = (image_array * 255).astype(np.uint8)

            # Convert to PIL image and save
            image = Image.fromarray(image_uint8).convert("RGB")
            image.save(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Input.jpg'))

        # Save GT heatmaps
        if gts.shape[0] == 2:
            fv_gt = gts[0].numpy()
            T_gt = gts[1].numpy()

            for arr, name, cbar in zip([fv_gt, T_gt], ["Fv_GT", "T_GT"], ["$Fv(r, z)$ [ppm]", "$T(r, z)$ [K]"]):
                r = np.linspace(0, 1, arr.shape[1])
                z = np.linspace(0, 1, arr.shape[0])
                title = f"{sample_number}_Heatmap of {name} ({sample_dir})"
                savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_{name}.jpg')
                heatmaps(r, z, arr, cbar, title, savefile)
        else:
            arr_gt = gts.numpy()
            r = np.linspace(0, 1, arr_gt.shape[1])
            z = np.linspace(0, 1, arr_gt.shape[0])
            if config.targetType == "T":
                 title = f"{sample_number}_Heatmap of T_GT ({sample_dir})"
                 cbarTitle = '$T(r, z)$ [K]'
                 savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_T_GT.jpg')
            elif config.targetType == "fv":
                title = f"{sample_number}_Heatmap of Fv_GT ({sample_dir})"
                cbarTitle = '$Fv(r, z)$ [ppm]'
                savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Fv_GT.jpg')
            heatmaps(r, z, arr_gt, cbarTitle, title, savefile)

    # Save predicted heatmaps
    if preds.shape[0] == 2:
        fv_pred = preds[0].numpy()
        T_pred = preds[1].numpy()

        for arr, name, cbar in zip([fv_pred, T_pred], ["Fv_Pred", "T_Pred"], ["$Fv(r, z)$ [ppm]", "$T(r, z)$ [K]"]):
            r = np.linspace(0, 1, arr.shape[1])
            z = np.linspace(0, 1, arr.shape[0])
            title = f"{sample_number}_Heatmap of {name} Epoch {epoch} ({sample_dir})"
            savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_E{epoch}_{name}.jpg')
            heatmaps(r, z, arr, cbar, title, savefile)
    else:
        arr_pred = preds[0].numpy()
        r = np.linspace(0, 1, arr_pred.shape[1])
        z = np.linspace(0, 1, arr_pred.shape[0])
        if config.targetType == "T":
            title = f"{sample_number}_Heatmap of T_Pred Epoch {epoch} ({sample_dir})"
            cbarTitle = '$T(r, z)$ [K]'
            savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_E{epoch}_T_Pred.jpg')
        elif config.targetType == "fv":
            title = f"{sample_number}_Heatmap of Fv_Pred Epoch {epoch} ({sample_dir})"
            cbarTitle = '$Fv(r, z)$ [ppm]'
            savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_E{epoch}_Fv_Pred.jpg')
        heatmaps(r, z, arr_pred, cbarTitle, title, savefile)

def save_error_heatmaps(outputs, gts, epoch, sample_id, inputs, out_dir, sample_name, config, loss):
    """
    Saves per-pixel absolute error heatmaps for fv and T.

    Parameters:
        outputs: (B, 2, H, W) - model predictions
        gts: (B, 2, H, W) - ground truth maps
        epoch: int - current epoch
        sample_id: str - index or tag for the sample
        inputs: (B, C, H, W) - original input images
        out_dir: str - base output directory
        sample_name: str - name or path of the sample
        config: config object with normalization info if needed
    """
    outputs = outputs.detach().cpu().numpy()
    gts = gts.detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy()

    batch_size = outputs.shape[0]
    for i in range(batch_size):
        pred = outputs[i]
        gt = gts[i]
        # input_img = inputs[i]

        if config.targetType == "both":
            error_fv = np.abs(pred[0] - gt[0])
            error_T = np.abs(pred[1] - gt[1])
        elif config.targetType == "fv":
            error_fv = np.abs(pred[0] - gt[0])
            error_T = None
        elif config.targetType == "T":
            error_T = np.abs(pred[0] - gt[0])
            error_fv = None
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
       
        if config.targetType == "both" or config.targetType == "fv":
            im1 = axs[0].imshow(error_fv, cmap='hot', vmin=0, vmax=1)
            axs[0].set_title(f"Error Heatmap - fv - {loss:.4f}")
            fig.colorbar(im1, ax=axs[0])
        

        
        
        if config.targetType == "both" or config.targetType == "T":
            im2 = axs[1].imshow(error_T, cmap='hot', vmin=0, vmax=1)
            axs[1].set_title(f"Error Heatmap - T - {loss:.4f}")
            fig.colorbar(im2, ax=axs[1])

        for ax in axs:
            ax.axis('off')

        # os.makedirs(os.path.join(out_dir, "error_maps"), exist_ok=True)
        plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, "error_maps", f"{epoch}_{sample_id}_{os.path.basename(sample_name)}_error.png"))
        plt.savefig(os.path.join(out_dir, f"{sample_id}_{os.path.basename(sample_name)}_E{epoch}_ErrorMaps.png"))
        plt.close()

# def saveheatmaps(outputs, gts, epoch, sample_number, inputs, heat_dir, sample_dir,config):
#     # On epoch 0, save gt heatmaps and the input image.
#     samp_folder = sample_dir[sample_dir.rfind('\\')+1:]
#     if epoch == 0 and (("stage1" in sample_number) or ("test" in sample_number)):
#         gts = gts[0].view(config.output_shape[0], config.output_shape[1])
#         fv_gt = gts.cpu().numpy()

#         r = fv_gt.shape[1]
#         z = fv_gt.shape[0]
#         title = f"{sample_number}_Heatmap of $Fv(r, z)$ GT ({sample_dir})"
#         cbarTitle = '$Fv(r, z)$ [ppm]'
#         savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_GT.jpg')
#         heatmaps(r, z, fv_gt, cbarTitle, title, savefile)

#         # Save input image
#         inputs = inputs[0].cpu().detach()
#         if inputs.ndimension() == 4:
#             inputs = inputs.squeeze(0)
#         to_pil = transforms.ToPILImage().Im
#         image = to_pil(inputs)
#         image.save(os.path.join(heat_dir, f'{sample_number}_{samp_folder}_Input.jpg'))
    
#     # Save prediction heatmaps for current epoch.
#     preds = outputs[0].view(config.output_shape[0], config.output_shape[1])
#     fv_pred = preds.cpu().numpy()

#     r = fv_pred.shape[1]
#     z = fv_pred.shape[0]
#     title = f"{sample_number}_Heatmap of $Fv(r, z)$ Epoch: {epoch}  ({sample_dir})"
#     cbarTitle = '$Fv(r, z)$ [ppm]'
#     savefile = os.path.join(heat_dir, f'{sample_number}_{samp_folder}_E{epoch}.jpg')
#     heatmaps(r, z, fv_pred, cbarTitle, title, savefile)