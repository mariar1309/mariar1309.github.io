# CS194-26 (CS294-26): Project 1 starter Python code

##########################################################   Set up   ###############################################################
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as transform

# name of the input file
imname = 'tobolsk.jpg'
# read in the image
im = skio.imread(imname)
# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

####################################################### L2 ##########################################################
# align the images
'''
Description: 
"The easiest way to align the parts is to exhaustively search over a window of possible displacements (say [-15,15] pixels), 
score each one using some image matching metric, and take the displacement with the best score. 
There is a number of possible metrics that one could use to score how well the images match. 
The simplest one is just the L2 norm also known as the Euclidean Distance which is simply sqrt(sum(sum((image1-image2).^2))) where the sum is taken over the pixel values"
'''
#reference_channel will always be B in our case
#channel_to_align is either R or G
def l2_alignment(reference_channel, channel_to_align, window = 15, init_displacement=(0,0)):
    # search over a window of possible displacements (say 15x15)
    #UPDATE: making it static here hurts the pyramid resolution laer, move to function arguments
    #window = 15
    # score each using L2 metric, take displacement with best score (the one closest to 0!)
    best_l2_score = float('inf')
    best_displacement = init_displacement

    crop = max(10, reference_channel.shape[0] // 20)
    reference_crop = reference_channel[crop:-crop, crop:-crop]

    for x in range(init_displacement[0] - window, init_displacement[0] + window + 1):
        for y in range(init_displacement[1] - window, init_displacement[1] + window + 1):

            #tried usng np.roll to test different alignemnts of R and G over B
            aligned_channel = np.roll(channel_to_align, (x, y), axis=(0, 1))
            aligned_channel_crop = aligned_channel[crop: - crop, crop: - crop]

            #l2 norm (aka euldian distance) = sqrt(sum(sum((image1-image2).^2)))
            #difference = aligned_channel - reference_channel
            difference = aligned_channel_crop - reference_crop
            root = np.sqrt(np.sum(difference ** 2))

            # the closer to 0, the better
            if root < best_l2_score:
                best_l2_score = root
                best_displacement = (x, y)

            #test
            #print(f"best displacement:  {best_displacement}, best_l2_score:  {best_l2_score}")
    #test
    print(f"best displacement:  {best_displacement}, best_l2_score:  {best_l2_score}")

        
    best_alignment = np.roll(channel_to_align, best_displacement, axis=(0, 1))
    return best_alignment, best_displacement

####################################################### NCC #######################################################
#trying out the NCC method too
# "Normalized Cross-Correlation (NCC), which is simply a dot product between two normalized vectors: (image1./||image1|| and image2./||image2||)."

def ncc_alignment(reference_channel, channel_to_align, window=15, init_displacement=(0,0)):
    #window = 15
    best_score = - float('inf')
    best_displacement = init_displacement

    #need to normalize the reference channel, as well as both aligning channels

    #normalise the ref channel first:
    crop = max(50, reference_channel.shape[0] // 10)
    reference_crop = reference_channel[crop: - crop, crop: - crop]
    #ncc for reference
    reference_mean = np.mean(reference_crop)
    reference_std = np.std(reference_crop)
    #reference_ncc = (reference_crop - reference_mean) / (reference_std)
    reference_ncc = (reference_crop - reference_mean) / (reference_std + 1e-8)

    for x in range(init_displacement[0] - window, init_displacement[0] + window + 1):
        for y in range(init_displacement[1] - window, init_displacement[1] + window + 1):

            aligned_channel = np.roll(channel_to_align, (x, y), axis=(0, 1))
            aligned_channel_crop = aligned_channel[crop: - crop, crop: - crop]
            
            #ncc for aligned_channel
            aligned_mean = np.mean(aligned_channel_crop)
            aligned_std = np.std(aligned_channel_crop)
            #aligned_ncc = (aligned_channel_crop - aligned_mean) / (aligned_std)
            aligned_ncc = (aligned_channel_crop - aligned_mean) / (aligned_std + 1e-8)


            #find ncc (dot product between reference and aligned layers)
            curr_ncc = np.sum(reference_ncc * aligned_ncc) / (reference_ncc.size)

            if curr_ncc > best_score:
                best_score = curr_ncc
                best_displacement = (x, y)

            #test
            #print(f"best displacement:  {best_displacement}, best_ncc_score:  {best_ncc_score}")
    #test
    print(f"best displacement:  {best_displacement}, best_ncc_score:  {best_score}")

    best_alignment = np.roll(channel_to_align, best_displacement, axis=(0, 1))
    return best_alignment, best_displacement

##################################################### Pyramid ##############################################################

#the ideaa is to reduce images in size, stacking lower resolution images on top f higher resolution ones, allowing for faster processing of smaller, lowres images before carrying it over to higher ones
#inputs: reference channel, channel aligning with (r or g), alignment function (l2 or ncc)

#pyramid window and total levels 
#window = 15
#levels = 3 #try 4 and 3

def pyramid(reference_channel, alignment_channel, alignment_function, window = 15, levels = 4):
    #yaaaayyy recursion 
    if levels == 0:
        print(f"{window} size at level 0 reached, end rec")
        aligned_channel, displacement = alignment_function(reference_channel, alignment_channel, window)
        return aligned_channel, displacement
    
    else:
        #make a lowpass pyramid of lower res layesrs , scaling by factor of 2 (wiki)

        min_frame = 20
        if reference_channel.shape[0] < min_frame or reference_channel.shape[1] < min_frame:
            aligned_channel, displacement = alignment_function(reference_channel, alignment_channel, window)
            return aligned_channel, displacement

        scale = 0.5
        downscale_reference = transform.rescale(reference_channel, scale, anti_aliasing=True, channel_axis=None)
        downscale_alignment = transform.rescale(alignment_channel, scale, anti_aliasing=True, channel_axis=None)

        new_window = max(1, window // 2)
        align_level_down, disp_level_down = pyramid(downscale_reference, downscale_alignment, alignment_function, new_window, levels-1)

        scale_disp = (disp_level_down[0] * 2, disp_level_down[1] * 2)

        test_align = np.roll(alignment_channel, scale_disp, axis=(0,1))

        smaller_window = 5
        best_alignment, best_displacement = alignment_function(reference_channel, test_align, smaller_window, init_displacement = (0,0))

        total_displacement = (scale_disp[0] + best_displacement[0], scale_disp[1] + best_displacement[1])

        return best_alignment, total_displacement

        


# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

### ag = align(g, b)
### ar = align(r, b)

# create a color image
print("align green channel:")
#ag, green_displacement = ncc_alignment(b, g, window=15)
ag, green_displacement = pyramid(b, g, ncc_alignment, window=50, levels=5)
print(f"final green displacement: {green_displacement}")

print("align red channel:")
#ar, red_displacement = ncc_alignment(b, r, window=15)
ar, red_displacement = pyramid(b, r, ncc_alignment, window=50, levels=5)
print(f"final red displacement: {red_displacement}")

im_out = np.dstack([ar, ag, b])

#fixing conversion issues
im_out = (im_out * 255).astype(np.uint8)

# save the image
fname = 'pyr_' + imname
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()
