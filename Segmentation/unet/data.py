from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import time
# import imageio
import skimage.transform as trans
import cv2 as cv
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing import image


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '¦', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def adjustData(img,mask,flag_multi_class,num_class, image_color_mode="rgb"):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # this line inverts the colors in mask
        # mask = 1 - mask
    # else:
    #     mask[mask > 0.5] = 1
    #     mask[mask <= 0.5] = 0
        # this line inverts the colors in mask
        # mask = 1 - mask
    # if image_color_mode == 'grayscale':
        # img = np.concatenate((img, img, img), axis=3)
    return (img,mask)



def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(512, 512), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,  # comment
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class, image_color_mode)
        yield (img, mask)


def load_image(img_path, height=100, width=100):

    img = image.load_img(img_path, target_size=(height, width))
    # img = image.load_img(img_path, target_size=(height, width), color_mode="grayscale")
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def load_img_generator(imgs_path, height, width):
    for img_path in imgs_path:
        yield load_image(img_path, height, width)


def test_single_img(model, img_path, out_path, height, width):
    img = load_image(img_path, height, width)
    pred = model.predict(img)
    pred_out = pred[0, :, :, :]
    # pred_out[pred_out >= 0.05] = 1
    # pred_out[pred_out < 0.05] = 0
    img_name = os.path.basename(img_path)

    # save_img(out_path + 'out_' + img_name, pred_out)
    out_img_path = os.path.join(out_path, img_name)
    save_img(out_img_path, pred_out)


def test_batch_imgs(model, batch_imgs_names, imgs_generator, out_path, height, width, gpus_num=1):
    # pred = model.predict(imgs_generator, batch_size=2 * gpus_num)
    pred = model.predict(imgs_generator, steps=4)
    for i in range(2*gpus_num):
        pred_out = pred[i, :, :, :]
        # pred_out[pred_out >= 0.05] = 1
        # pred_out[pred_out < 0.05] = 0
        img_name = os.path.basename(batch_imgs_names[i])

        # save_img(out_path + 'out_' + img_name, pred_out)
        out_img_path = os.path.join(out_path, img_name)
        save_img(out_img_path, pred_out)


def test_model(model, test_path, height, width, out_path=None):
    test_files = glob.glob(test_path)
    length = len(test_files)
    printProgressBar(0, length, prefix='Segmenting:', suffix='Complete', length=50)
    for i, test_file in enumerate(test_files):
        if out_path:
            test_single_img(model, test_file, out_path, height, width)
        else:
            test_single_img(model, test_file, test_path[:-1], height, width)

        # time.sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, length, prefix='Segmenting:', suffix='Complete', length=50)


# Yields successive 'n' sized chunks from list 'list_name'
def create_chunks(list_name, n):
    for i in range(0, len(list_name), n):
        yield list_name[i:i + n]


def test_model_parallel(model, test_path, height, width, out_path=None, gpus_num=1):
    test_files = glob.glob(test_path)
    length = len(test_files)
    printProgressBar(0, length, prefix='Segmenting:', suffix='Complete', length=50)
    test_files_chunks = list(create_chunks(test_files, 2 * gpus_num))
    for i, test_files in enumerate(test_files_chunks):
        imgs_generator = load_img_generator(test_files, height, width)
        if out_path:
            test_batch_imgs(model, test_files_chunks[i], imgs_generator, out_path, height, width, gpus_num)
        else:
            test_batch_imgs(model, test_files_chunks[i], imgs_generator, test_path[:-1], height, width, gpus_num)

        # time.sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, length, prefix='Segmenting:', suffix='Complete', length=50)


def testGenerator(test_path,num_image = 30,target_size = (512, 512),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        # img = io.imread(os.path.join(test_path,"%d.jpg"%i),as_gray = as_gray)
        # img = img / 255
        # img = trans.resize(img,target_size)
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        # img = np.reshape(img,(1,)+img.shape)

        img = image.load_img(os.path.join(test_path,"%d.png"%i), target_size=target_size)
        # img = image.load_img(os.path.join(test_path, "%d.jpg" % i), target_size=target_size, color_mode="grayscale")
        img_tensor = image.img_to_array(img)  # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor,
                                    axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.
    yield img_tensor


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        # img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item
        # print("img shape:")
        # print(img.shape)
        # print("img: ")
        # print(img)
        save_img(os.path.join(save_path, "%d_predict.jpg" % i), img)
        # img_uint8 = img.astype(np.uint8)
        # and then
        # imageio.imwrite('filename.jpg', img_uint8)
        # imageio.imwrite(os.path.join(save_path, "%d_predict.png" % i), img_uint8)
        # cv.imwrite(os.path.join(save_path, "%d_predict.png" % i), img_uint8)
