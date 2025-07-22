from PIL import Image

# Open the TIFF image
tif_image = Image.open('ExpandedSpheres_004.tif')

# Convert and save as JPEG
tif_image.convert('RGB').save('spheres.jpg', 'JPEG')