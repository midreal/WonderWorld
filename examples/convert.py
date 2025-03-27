from PIL import Image
import glob
import os

# Get all jpg files in the images directory
jpg_files = glob.glob('images/*.jpg')

# Convert each jpg file to png
for jpg_file in jpg_files:
    # Open the jpg image
    image = Image.open(jpg_file)
    
    # Create the output filename by replacing .jpg with .png
    png_file = os.path.splitext(jpg_file)[0] + '.png'
    
    # Save as PNG
    image.save(png_file)
    print(f'Converted {jpg_file} to {png_file}')