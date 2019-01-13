import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import create_model
from PIL import Image
from glob import glob

image_files = glob('./test_images/*.*')

model = keras.models.load_model('trained_model.h5')

def load_image( infilename ) :
    img = Image.open( infilename )
    width, height = img.size
    scale = height * 1.0 / 28
    width = int(width / scale)
    #img = img.resize([width, 28], Image.ANTIALIAS)
    img = img.resize([28, 28], Image.ANTIALIAS)
    print img.size

    img.load()
    img = np.asarray( img, dtype="int32" )

    return img


if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        # image = np.array(Image.open(image_file).convert('RGB'))
        img = load_image(image_file)
        t = time.time()
        result = model.predict(img)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        print result

