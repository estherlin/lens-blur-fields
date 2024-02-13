import torch
import numpy as np

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2, axis=(0,1))
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def PSNR_patterns(original, compressed):
    mse = np.mean((original - compressed) ** 2, axis=(0,1,2))
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def PSNR_pattern_stack(original, compressed):
    mse = np.mean((original - compressed) ** 2, axis=(2,3))
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def find_focus_p(sharp_checkerboard, blurred_checkerboard, positions, row=600, col=700):
    psnrs = np.zeros((len(positions)), np.float32)

    for i in range(len(positions)):
        psnr_i = PSNR(sharp_checkerboard[0,0,i, row:-row,col:-col, 0], blurred_checkerboard[0,0,i, row:-row,col:-col, 0])
        psnrs[i] = psnr_i
    return np.argmax(psnrs), psnrs

def rmnan(tensor):
    tensor[torch.isnan(tensor)] = 0
    tensor[torch.isinf(tensor)] = 0
    return tensor


def antialiased_circ(N, rad):
  # Make a kernel
  height, width = N, N # kernel size
  mid = height//2 # middle of kernel

  # Make a circular mask
  kernel_size = height
  kernel_radius = (kernel_size) // 2
  cen_x = 0
  cen_y = 0

  # make a grid
  x, y = np.ogrid[-kernel_radius:kernel_radius, -kernel_radius:kernel_radius]
  dist = (np.square(x-cen_y) + np.square(y-cen_x))**0.5 # shape (kernel_size, kernel_size)

  # Make second column the filter
  radii = np.array([rad])
  antialiased_kern = 1 - (dist - radii).clip(0,1) # shape (num_radii, kernel_size, kernel_size)

  if N % 2 == 1:
    temp = np.zeros((N, N))
    temp[0:N-1, 0:N-1] = antialiased_kern
    antialiased_kern = temp
  return antialiased_kern


def aliased_circ(N, rad):

  aliased_kern = np.zeros((N, N))
  y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
  mask = x**2 + y**2 <= rad**2
  aliased_kern[mask] = 1
  return aliased_kern


def psnr_torch(img1, img2, device, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2, dim=(0,2,3))
    return 10 * torch.log10(max_val**2 / mse)

def psnr_bayer(img1, img2, device, max_val=1.0):
    num_patterns, num_p, h, w = img1.shape
    im1, im2 = torch.zeros((num_patterns, num_p, h//2, w//2, 4), device=device), torch.zeros((num_patterns, num_p, h//2, w//2, 4), device=device)
    im1[...,0] = img1[...,0::2,0::2]
    im1[...,1] = img1[...,0::2,1::2]
    im1[...,2] = img1[...,1::2,0::2]
    im1[...,3] = img1[...,1::2,1::2]
    im2[...,0] = img2[...,0::2,0::2]
    im2[...,1] = img2[...,0::2,1::2]
    im2[...,2] = img2[...,1::2,0::2]
    im2[...,3] = img2[...,1::2,1::2]
    return psnr_torch(im1, im2, device, max_val=max_val)


def dp_disk(kersig, kersize):
    """ Punnaparath's dp model"""
    # Compute the radius of the circle
    radius = np.abs(kersig)
    # make an antialisased circle
    refcirc = antialiased_circ(radius, kersize)
    
    # Create an array of distances from the center of the circle
    dist_array = np.linspace(0, 2*radius+1, 2*radius+1+1)
    # Initialize the output kernel with zeros
    diskker = np.zeros((kersize, kersize))

    # Iterate over the distances and sum the translated circles
    for i in dist_array:
        # Shift the reference circle by i pixels along the x-axis
        shifted_refcirc = np.roll(refcirc, int(np.sign(kersig) * i), axis=1)
        diskker += np.multiply(shifted_refcirc, refcirc)
    # Normalize the output kernel
    kerout = diskker / np.sum(diskker)
    return kerout, refcirc

def blur_rad(A, distance, S, F):
    "Calculate the blur radius for a given distancefrom thin lens formula"
    return A*np.abs(distance - S) * F / (distance * (S-F))