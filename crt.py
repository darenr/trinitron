from PIL import Image
import numpy as np
import random

def apply_spherical_aberration(img_array, width, height, strength=0.1):
    # Create coordinate grids
    y, x = np.indices((height, width))
    
    # Center coordinates
    center_x = width / 2
    center_y = height / 2
    
    # Calculate distance from center
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Normalize distance
    norm_dist = dist / max_dist
    
    # Calculate displacement based on spherical aberration
    displacement = norm_dist**2 * strength
    
    # Create new coordinate mapping
    new_x = x + (x - center_x) * displacement
    new_y = y + (y - center_y) * displacement
    
    # Ensure coordinates stay within bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    # Create output array
    output = np.zeros_like(img_array)
    
    # Interpolate pixels to new positions
    for channel in range(img_array.shape[2]):
        # Use bilinear interpolation from the original array
        for i in range(height):
            for j in range(width):
                x0, y0 = int(new_x[i, j]), int(new_y[i, j])
                x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
                
                # Get surrounding pixel values
                f00 = img_array[y0, x0, channel]
                f01 = img_array[y1, x0, channel]
                f10 = img_array[y0, x1, channel]
                f11 = img_array[y1, x1, channel]
                
                # Calculate weights
                wx = new_x[i, j] - x0
                wy = new_y[i, j] - y0
                
                # Bilinear interpolation
                value = (f00 * (1 - wx) * (1 - wy) +
                        f10 * wx * (1 - wy) +
                        f01 * (1 - wx) * wy +
                        f11 * wx * wy)
                
                output[i, j, channel] = value
    
    return output

def apply_trinitron_effect(input_path, output_path):
    # Open the image
    img = Image.open(input_path)
    
    # Convert to RGBA for processing
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Get image dimensions
    width, height = img.size
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Create output array
    output_array = np.copy(img_array)
    
    # Simulate Trinitron RGB pattern (horizontal)
    for y in range(height):
        pixel_pos = y % 3
        if pixel_pos == 0:  # Red phosphor
            output_array[y, :, 1] = output_array[y, :, 1] * 0.7  # Reduce green
            output_array[y, :, 2] = output_array[y, :, 2] * 0.7  # Reduce blue
        elif pixel_pos == 1:  # Green phosphor
            output_array[y, :, 0] = output_array[y, :, 0] * 0.7  # Reduce red
            output_array[y, :, 2] = output_array[y, :, 2] * 0.7  # Reduce blue
        else:  # Blue phosphor
            output_array[y, :, 0] = output_array[y, :, 0] * 0.7  # Reduce red
            output_array[y, :, 1] = output_array[y, :, 1] * 0.7  # Reduce green
    
    # Add vertical scanlines
    for x in range(width):
        if x % 2 == 0:
            output_array[:, x, :] = output_array[:, x, :] * 0.85
    
    # Add subtle noise
    noise = np.random.normal(0, 10, output_array.shape)
    output_array = output_array + noise
    output_array = np.clip(output_array, 0, 255)
    
    # Apply spherical aberration
    output_array = apply_spherical_aberration(output_array, width, height, strength=0.1)
    
    # Slight brightness adjustment
    output_array = output_array * 1.05
    output_array = np.clip(output_array, 0, 255)
    
    # Convert back to image
    output_img = Image.fromarray(output_array.astype('uint8'))
    
    # Convert to RGB for JPEG compatibility
    output_img = output_img.convert('RGB')
    
    # Save the result
    output_img.save(output_path)

# Example usage
if __name__ == "__main__":
    input_image = "input.jpg"
    output_image = "output_trinitron.jpg"
    
    try:
        apply_trinitron_effect(input_image, output_image)
        print(f"Image processed successfully and saved as {output_image}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")