#I need just one chanel of the three-chanel glass image for the website formatting

import numpy as np
import skimage as sk
import skimage.io as skio

imname = 'monastery.jpg'  
im = skio.imread(imname)

im = sk.img_as_float(im)

height = np.floor(im.shape[0] / 3.0).astype(int)

channel_number = 2  #0b. 1g. 2r
channel_names = ['blue', 'green', 'red']
selected_channel = im[channel_number * height : (channel_number + 1) * height]

if selected_channel.dtype == np.float32 or selected_channel.dtype == np.float64:
    selected_channel = (selected_channel * 255).astype(np.uint8)
elif selected_channel.dtype == np.uint16:
    selected_channel = (selected_channel / 256).astype(np.uint8)

output_filename = f'{channel_names[channel_number]}_channel_bw.jpg'
skio.imsave(output_filename, selected_channel)

