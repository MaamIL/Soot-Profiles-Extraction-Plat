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
    contour = plt.contourf(R, Z, fvORt, levels=50, cmap='inferno')
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
    # Optional image save (only once)
    # if epoch == 0 and (("stage1" in sample_number) or ("test" in sample_number)):
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