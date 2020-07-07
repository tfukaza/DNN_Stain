import staintools
import numpy as np

METHOD = 'vahadane'
STANDARDIZE_BRIGHTNESS = True
RESULTS_DIR = './results/'
EXTRACT_TYPE = 0

images=[]

# Read data
for i in range(1,6):
    images.append(staintools.read_image('./data/i' + str(i) + '.png'))

# Plot
# Images will be normalized to the 'target' image
titles = ["Target"] + ["Original"] * (len(images) - 1)
staintools.plot_image_list(images, width=len(images), title_list = titles)

# Brightness standardization
if STANDARDIZE_BRIGHTNESS:
    # Standarize brightness
    images = list(map(staintools.LuminosityStandardizer.standardize, images))

    # Plot
    titles = ["Target Standarized"] + ["Original Standarized"] * (len(images) - 1)
    staintools.plot_image_list( images, len(images), title_list=titles,
                                save_name=RESULTS_DIR + 'original-images-standarized.png', show=0)

# Stain normalization
# Normalize to stain of first image
normalizer = staintools.StainNormalizer(method=METHOD)
normalizer.fit(images[0])
S = normalizer.stain_matrix_target
images = [images[0]] + list(map(normalizer.transform, images[1:]))

# Plot
titles = ["Target"] + ["Stain Normalized"] * (len(images) - 1)
staintools.plot_image_list( images, width=len(images), title_list=titles,
                            save_name = RESULTS_DIR + 'stain-normalized-images.png', show=0)

def stain_feature(images=[], feature=1):
    l = []
    for i, I in enumerate(images):
        
        # Create an instance of the normalizer class for this image
        tmp = staintools.StainNormalizer(method=METHOD)
        tmp.fit(I)
       
        C = tmp.target_concentrations            # Get concentration matrix
        
        C = np.insert(C, 2, C[:,feature], axis=1)
        C = np.insert(C, 2, C[:,feature], axis=1)
        C = np.insert(C, 2, C[:,feature], axis=1)
        #print(C)
        C = C[:,2:] 
        #print(C)
        l.append(255 * np.exp(-1 * C.reshape(I.shape).astype(np.uint8)))

    # Plot
    titles = ["Extract feature"] * (len(l))
    staintools.plot_image_list( l, width=len(l), title_list=titles,
                                save_name = RESULTS_DIR + 'feature' + str(feature) + '-extract-images.png', show=0)

stain_feature(images, 0)
stain_feature(images, 1)

# # Stain Augmentation
# augmentor = staintools.StainAugmentor(method=METHOD, sigma1=0.4, sigma2=0.4)
# augmentor.fit(images[0])
# augmented_images = []
# for _ in range(10):
#     augmented_img = augmentor.pop()
#     augmented_images.append(augmented_img)

# # Plot 
# titles = ["Augmented"] * 10
# staintools.plot_image_list( augmented_images, width=len(augmented_images), title_list=titles,
#                             save_name = RESULTS_DIR + 'stain-augmented-images.png', show=0)


