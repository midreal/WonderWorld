from PIL import Image
import numpy as np

def feather(mask_path, feather_width=80):
    """
    Create a feathered white feather extending from white regions into black regions.
    
    Args:
        mask_path (str): Path to the input mask image
        feather_width (int): Width of the feather in pixels
        
    Returns:
        None: Saves the result as mask_feather.png
    """
    # Load mask image
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)
    
    # Create distance field arrays
    dist_field = np.zeros_like(mask_array, dtype=np.float32)
    output = mask_array.copy().astype(np.float32)
    height, width = mask_array.shape
    
    # First pass: mark boundary pixels and initialize distance field
    boundary_pixels = []
    for y in range(1, height-1):
        for x in range(1, width-1):
            if mask_array[y, x] > 128:  # White pixel
                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if mask_array[ny, nx] < 128:  # Found black neighbor
                            boundary_pixels.append((y, x))
                            break
                    else:
                        continue
                    break
    
    # Second pass: compute distance field using boundary pixels
    sigma = feather_width / 2.0  # Increased base sigma for stronger effect
    for y, x in boundary_pixels:
        # Create local Gaussian kernel
        kernel_size = int(feather_width * 1.5)
        y_indices = np.arange(max(0, y - kernel_size), min(height, y + kernel_size + 1))
        x_indices = np.arange(max(0, x - kernel_size), min(width, x + kernel_size + 1))
        Y, X = np.meshgrid(y_indices - y, x_indices - x, indexing='ij')
        
        # Compute Gaussian values
        distances = np.sqrt(X*X + Y*Y)
        gaussian = np.exp(-0.5 * (distances/sigma)**2)
        
        # Update distance field
        local_mask = mask_array[y_indices[:, None], x_indices] < 128
        gaussian = gaussian * local_mask  # Only affect black regions
        
        # Accumulate values using maximum
        dist_field[y_indices[:, None], x_indices] = np.maximum(
            dist_field[y_indices[:, None], x_indices],
            gaussian
        )
    
    # Final pass: apply feathering with additional smoothing
    # First convert to float for better precision during smoothing
    output = np.where(mask_array > 128, 255, dist_field * 255).astype(np.float32)
    
    # Create Gaussian kernel for smoothing
    smooth_sigma = feather_width / 2.5  # Increased sigma for stronger smoothing
    kernel_size = int(smooth_sigma * 3) * 2 + 1  # Larger kernel for wider effect
    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    smooth_kernel = np.exp(-(x*x + y*y)/(2*smooth_sigma*smooth_sigma))
    smooth_kernel = smooth_kernel / smooth_kernel.sum()  # Normalize
    
    # Boost the transition intensity
    dist_field = np.power(dist_field, 0.7)  # Boost transition intensity
    
    # Apply smoothing only to transition areas
    transition_mask = (output > 0) & (output < 255)
    if transition_mask.any():
        # Pad the image to handle edges properly
        pad_size = kernel_size // 2
        padded = np.pad(output, pad_size, mode='reflect')
        
        # Apply convolution only to transition areas
        smoothed = output.copy()
        from scipy.ndimage import convolve
        smoothed[transition_mask] = convolve(
            padded, 
            smooth_kernel, 
            mode='constant'
        )[pad_size:-pad_size, pad_size:-pad_size][transition_mask]
        
        # Blend smoothed result with original based on distance to edges
        blend_factor = np.clip((output - 128) / 128, 0, 1)
        output = blend_factor * output + (1 - blend_factor) * smoothed
    
    # Convert back to uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Save result
    # Ensure all white areas remain white
    output[mask_array > 128] = 255
    
    # Create and save the featherd mask image
    feather_img = Image.fromarray(output)
    return feather_img

def combine(img_path, mask_path):
    """
    Combine an image with a mask to create transparency.
    
    Args:
        img_path (str): Path to the input image
        mask_path (str): Path to the mask image (white = transparent)
        
    Returns:
        None: Saves the result as img_w_mask.png
    """
    # Load images
    img = Image.open(img_path).convert('RGBA')
    mask = Image.open(mask_path).convert('L')
    
    # Convert to numpy arrays
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    # Create alpha channel (255 - mask because white should be transparent)
    alpha = 255 - mask_array
    
    # Create RGBA array
    height, width = mask_array.shape
    rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_array[..., :3] = img_array[..., :3]  # Copy RGB channels
    rgba_array[..., 3] = alpha  # Set alpha channel
    
    # Save final image
    final_img = Image.fromarray(rgba_array, 'RGBA')
    return final_img

if __name__ == "__main__":
    # Demo usage
    # Create featherd mask
    feather("moon_mask.png", feather_width=50)
    
    # Combine original image with featherd mask
    combine("moon_modified.png", "mask_feather.png")
