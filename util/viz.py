import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF



COLOUR_MAP = {
    0: "red", 1: "green", 2: "orange",3: "blue",
}
COLOUR_CODES = {
    0: "r", 1: "g1", 2: "g2", 3: "b",
}

def plot_stats(losses, psnrs, lens_positions, save_path):
    fig, ax = plt.subplots(1, 2, figsize=[18, 4])
    ax[0].plot(losses.keys(), losses.values())
    ax[0].set_title("MSE vs itr")
    ax[0].set_yscale('log')
    for i in range(4):
        ax[1].plot(lens_positions, psnrs[...,i].mean(axis=0), c=COLOUR_MAP[i], label=COLOUR_CODES[i])
        ax[1].fill_between(
            lens_positions, psnrs[...,i].min(axis=0), psnrs[...,i].max(axis=0), alpha=0.2, color=COLOUR_MAP[i])
        ax[1].set_title("PSNRS per lens pos")
    ax[1].legend()
    plt.savefig(f"{save_path}/stats.png")
    plt.close()

def write_4D_image(image, height, width, filename):
	# Save the synthetic image
	image = image.cpu().detach().numpy()
	rgb_synthetic = np.zeros((height, width, 3), dtype=np.float32)
	rgb_synthetic[...,0] = image[...,0]
	rgb_synthetic[...,1] = 0.5*(image[...,1] + image[...,2])
	rgb_synthetic[...,2] = image[...,3]
	rgb_synthetic /= np.max(rgb_synthetic)
	temp = rgb_synthetic
	rgb_synthetic = torch.from_numpy(np.transpose(rgb_synthetic, (2, 0, 1)))
	TF.to_pil_image(rgb_synthetic).save(f"{filename}.png")
	return temp

def plot_grid(sharp_cut, blur_cut, estimated, kern, save_path, pad=30):
    for pat in range(sharp_cut.shape[0]):

        fig, ax = plt.subplots(4, 4, figsize=[20, 18])
        for i in range(4):
            cmax = max( blur_cut[pat,...,i].max(), estimated[pat,...,i].max())
            cmin = min( blur_cut[pat,...,i].min(), estimated[pat,...,i].min())

            im0 = ax[i,0].imshow(sharp_cut[pat,...,i])
            ax[i,0].set_title(f"Sharp")
            fig.colorbar(im0, ax=ax[i,0])
            im1 = ax[i,1].imshow(blur_cut[pat,...,i], vmax=cmax, vmin=cmin)
            ax[i,1].set_title(f"Blur")
            fig.colorbar(im1, ax=ax[i,1])
            im2 = ax[i,2].imshow(estimated[pat,...,i], vmax=cmax, vmin=cmin)
            ax[i,2].set_title(f"estimated")
            fig.colorbar(im2, ax=ax[i,2])
            im3 = ax[i,3].imshow(np.squeeze(kern)[..., pad:-pad, pad:-pad,i])
            ax[i,3].set_title(f"kernel")
            fig.colorbar(im3, ax=ax[i,3])
        fig.savefig(f"{save_path}-pattern-{pat}.png")
        plt.close()


def plot_kernel_gif(kerns, save_path):
    # Make a gif to store results for each lens position
    gif = imageio.get_writer(f"{save_path}.gif",mode='I')
    for p in range(kerns.shape[0]):
        frame = np.zeros((kerns.shape[1], kerns.shape[2]*2))
        frame[:,:kerns.shape[2]] = kerns[p,:,:,0]
        frame[:,kerns.shape[2]:] = kerns[p,:,:,1]
        frame = frame/np.max(frame)
        gif.append_data(np.uint8(frame*255.))
    gif.close()

