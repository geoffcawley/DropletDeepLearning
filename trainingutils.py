
import config
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.data import AUTOTUNE
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadDataset(csvpath, imagespath, dimensions=(0,0)):
    # load the contents of the CSV annotations file
    print(f"[INFO] loading dataset\nCSV: {csvpath}\nImages: {imagespath}")
    rows = open(csvpath).read().strip().split("\n")
    images = []
    targets = []
    filenames = []
    targetDict = dict()
    # loop over the rows
    (h, w) = dimensions
    for row in rows:
        # break the row into the filename and bounding box coordinates
        row = row.split(",")
        filename = row[0]
        # derive the path to the input image, load the image (in OpenCV
        # format), and grab its dimensions
        imagePath = filename
        if h == 0:
            image = cv2.imread(imagePath)
            (h, w) = image.shape[:2]

        target = list(map(lambda x: float(x)/w, row[1:]))
        # print(target)
        # input()

        # print(os.path.basename(filename))
        # input()
        targetDict[os.path.basename(filename)] = target

        # modelimg = cv2ToKerasImg(image, size=(224, 224))

        # update our list of data, targets, and filenames
        # images.append(modelimg)
        # targets.append(target)
        filenames.append(os.path.basename(filename))

    # print(targetDict)
    for root, dirs, files in os.walk(imagespath):
        for name in sorted(files):
            if name.endswith('.jpg'):
                targets.append(targetDict[name])
                
                # Show image for debugging
                # print(os.sep.join([root, name]))
                # dispimg = cv2.imread(os.sep.join([root, name]))
                # c = list(map(lambda x: int(x*h), targetDict[name]))
                # cv2.circle(dispimg, (c[0], c[1]), c[2], (0,255,0), 3)
                # dispimg = cv2.resize(dispimg, (600,600))
                # cv2.imshow('cb2c image', dispimg)
                # key = cv2.waitKey(0)
                # if key == 27:
                #     return
    # print(targets)           
    print(f'loaded {len(targets)} targets')
    return images, targets, filenames

def trainmodel(imagepath, targets, numOutputs, modelPath, plotPath, numepochs=25):

    batch_size = config.BATCH_SIZE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        imagepath,
        labels=targets,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        imagepath,
        labels=targets,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=batch_size)

    for image_batch, labels_batch in val_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))
    
    
    # normalize training and validation data
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    # visually verify dataset
    # validateDs(train_ds)
    # load the VGG16 network, ensuring the head FC layers are left off
    vgg = VGG16(weights="imagenet", include_top=False,
                input_tensor=Input(shape=(224, 224, 3)))

    # freeze all VGG layers so they will *not* be updated during the
    # training process
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)

    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(numOutputs, activation="sigmoid")(bboxHead)

    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)

    # initialize the optimizer, compile the model, and show the model
    # summary
    opt = Adam(learning_rate=config.INIT_LR)
    model.compile(loss='mae', optimizer=opt)
    print(model.summary())

    # train the network for bounding box regression
    print("[INFO] training droplet detector")
    N = numepochs
    # H = model.fit(
    #     trainImages, trainTargets,
    #     validation_data=(testImages, testTargets),
    #     batch_size=config.BATCH_SIZE,
    #     epochs=N,
    #     verbose=1)
    H = model.fit(
    	train_ds,
    	validation_data=val_ds,
    	# batch_size=config.BATCH_SIZE,
    	epochs=N,
    	verbose=1)

    # serialize the model to disk
    print("[INFO] saving object detector model...")
    model.save(modelPath)

    # plot the model training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)

def cv2ToKerasImg(cv2img, size=(0,0)):
    if size == (0,0):
        size = cv2img.shape[:2]
    # print(size)
    # input()
    retimg = cv2.resize(cv2img, dsize=size)
    retimg = np.asarray(retimg).astype(float)
    # retimg = retimg / 255.0
    
    # retimg = np.expand_dims(retimg, axis=0)
    return retimg

def getBoundingSquare(bbox, h, w):
    (bbx1, bby1, bbx2, bby2) = bbox
    (bcx, bcy) = ((bbx1+bbx2)/2, (bby1+bby2)/2)
    halfside = max([bbx2-bbx1, bby2-bby1])/2
    (bsx1, bsy1, bsx2, bsy2) = (int(bcx-halfside),
                                int(bcy-halfside), int(bcx+halfside), int(bcy+halfside))
    if bsx1 < 0:
        bsx2 = int(2*halfside)
        bsx1 = 0
    if bsy1 < 0:
        bsy2 = int(2*halfside)
        bsy1 = 0
    if bsx2 >= w:
        bsx1 -= (bsx2-(w-1))
        bsx2 = w-1
    if bsy2 >= h:
        bsy1 -= (bsy2-(h-1))
        bsy2 = h-1
        
    return bsx1, bsy1, bsx2, bsy2

def cv2ToKerasImg(cv2img, size=(0, 0)):
    if size == (0, 0):
        size = cv2img.shape[:2]
    # print(size)
    # input()
    retimg = cv2.resize(cv2img, dsize=size)
    retimg = np.asarray(retimg).astype(float)
    # retimg = retimg / 255.0

    # retimg = np.expand_dims(retimg, axis=0)
    return retimg

def kerasToCv2Img(kerasimg, size=(0,0)):
    image = img_to_array(image)
    return image

def validateDs(ds):
    for images, labels in ds.take(1):
        for image, label in zip(images.numpy(), labels.numpy()):
            print("Image shape: ", image.shape)
            print("Label: ", label)
            image = img_to_array(image)
            label = list(map(lambda x: int(x*224), label))
            print(np.min(image), np.max(image))
            cv2.circle(image, (label[0], label[1]), label[2], (0,255,0), 2)
            cv2.imshow("dataset pipeline image", image)
            key = cv2.waitKey(0)
            if key == 27:
                raise Exception("ESC")
            
def forceTensorflowToUseGpu():
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print(os.getenv('TF_GPU_ALLOCATOR'))
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def getcb2cImg(image, box, debugmode=False):
    (h, w) = map(lambda x: int(x/2), image.shape[:2])
    cbimage = np.zeros((h, w, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    cbimage[:] = (255, 255, 255)
    cbimage[0:(box[3]-box[1]), 0:(box[2]-box[0])] = image[box[1]:box[3], box[0]:box[2]].copy()
    
    # show image
    if debugmode is True:
        cv2.imshow("cb2c image", cbimage)
        key = cv2.waitKey(0)
        if key == 27:
            exit()
    return cbimage

def getModelInput(image, sidelength):
    mi = cv2.resize(image, dsize=(sidelength, sidelength))
    mi = np.asarray(mi)
    mi = mi / 255.0
    mi = np.expand_dims(mi, axis=0)
    return mi