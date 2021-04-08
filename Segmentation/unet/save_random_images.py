import shutil, random, os
dirpath = '/home/matanr/MLography/Segmentation/unet/data/squares_256'
destDirectory = '/home/matanr/MLography/Segmentation/unet/data/120_squares_256'

filenames = random.sample(os.listdir(dirpath), 120)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    dstpath = os.path.join(destDirectory, fname)
    shutil.copyfile(srcpath, dstpath)