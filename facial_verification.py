import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
import tarfile
import random
import glob
import shutil

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


# Create Paths for positive, negative and Anchor Images
POSITIVE_PATH = os.path.join("data", "positive")
NEGATIVE_PATH = os.path.join("data", "negative")
ANCHOR_PATH = os.path.join("data", "anchor")


# Create directories for positive and anchor images
if os.path.isdir(POSITIVE_PATH) is False:
    os.makedirs(POSITIVE_PATH)
    os.makedirs(ANCHOR_PATH)

# download link for lfw dataset
# download "All images as gzipped tar file"
# url = http://vis-www.cs.umass.edu/lfw/#download

# Move the lfw.tgz file to the project directory

# extracting the lfw tar gz file
if os.path.isdir('lfw') is False:
    file = tarfile.open('lfw.tgz')
    file.extractall()
    file.close()


# Create negative directory
if os.path.isdir(NEGATIVE_PATH) is False:
    os.makedirs(NEGATIVE_PATH)
    # Moving images from lfw folder to negative folder
    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEGATIVE_PATH, file)
            os.replace(EX_PATH, NEW_PATH)


# Accessing the Web cam
cap = cv2.VideoCapture(0)

i = j = 0
while cap.isOpened():
    ret, frame = cap.read()

    # creating 250*250 frames
    frame = frame[120:370, 200:450, :]

    # Press a to collect anchor images
    # Collect anchors -- make sure to collect atleast 200 anchor images
    if cv2.waitKey(1) & 0xFF == 97:
        # create the unique file path
        imgname = os.path.join(ANCHOR_PATH, f'anchor_img{j}.jpg')
        cv2.imwrite(imgname, frame)
        j = j + 1

    # Press p to collect positive images
    # Collect positives -- make sure to collect atleast 200 positive images
    if cv2.waitKey(1) & 0xFF == 112:
        # create the unique file path
        imgname = os.path.join(POSITIVE_PATH, f'positive_img{i}.jpg')
        cv2.imwrite(imgname, frame)
        i = i + 1

    cv2.imshow('Image Collection', frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


# Taking 200 images from each directory
anchor = tf.data.Dataset.list_files(ANCHOR_PATH + '\*.jpg').take(200)
positive = tf.data.Dataset.list_files(POSITIVE_PATH + '\*.jpg').take(200)
negative = tf.data.Dataset.list_files(NEGATIVE_PATH + '\*.jpg').take(200)


# Preprocessing the data
def preprocess(file_path):
    # Read the images
    read_img = tf.io.read_file(file_path)
    # Convert images to tensors
    img = tf.io.decode_jpeg(read_img)
    # Resizing the images to 105 * 105
    img = tf.image.resize(img, (105, 105))
    # Normalizing the images
    img = img / 255.0
    return img


# Creating training data
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


# Preprocessing training data
def preprocess_data(input_img, val_img, label):
    return (preprocess(input_img), preprocess(val_img), label)

# Mapping the data
data = data.map(preprocess_data)
# Shuffling the data
data = data.shuffle(buffer_size=1024)
# Creating the batch of 16
data = data.batch(16)


# Create the model
def create_model():
    inp = Input(shape=(105, 105, 3), name='input_image')
    conv1 = Conv2D(64, (10, 10), activation='relu')(inp)
    max1 = MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = Conv2D(128, (7, 7), activation='relu')(max1)
    max2 = MaxPooling2D((2, 2), padding='same')(conv2)
    conv3 = Conv2D(128, (4, 4), activation='relu')(max2)
    max3 = MaxPooling2D((2, 2), padding='same')(conv3)
    conv4 = Conv2D(256, (4, 4), activation='relu')(max3)
    flat1 = Flatten()(conv4)
    dense1 = Dense(4096, activation='sigmoid')(flat1)

    return Model(inputs=[inp], outputs=[dense1], name='model')


model = create_model()


# creating L1Dist class to calculate the difference value between two images
class L1Distance(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_value, valid_value):
        return tf.math.abs(input_value - valid_value)


def create_siamese_model():
    input_image = Input(name='input_image', shape=(105, 105, 3))
    validation_image = Input(name='validation_image', shape=(105, 105, 3))

    siamese_layer = L1Distance()
    distances = siamese_layer(model(input_image), model(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='siameseNetwork')


siamese_model = create_siamese_model()


binary_loss = tf.losses.BinaryCrossentropy()
optimizer = tf.optimizers.Adam(1e-4)


# Training the model
def train(data, epochs):
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}/{epochs}')
        progbar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            with tf.GradientTape() as tape:
                x = batch[:2]
                y = batch[2]

                y_pred = siamese_model(x, training=True)
                loss = binary_loss(y, y_pred)
            progbar.update(idx + 1)

            grad = tape.gradient(loss, siamese_model.trainable_variables)
            optimizer.apply_gradients(zip(grad, siamese_model.trainable_variables))


if os.path.isfile('siamesemodel.h5') is False:
    train(data, epochs=50)
    siamese_model.save('siamesemodel.h5')


final_model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Distance':L1Distance, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy()})

# Creating the path for validation and input images
VAL_PATH = os.path.join('app_data', 'verification_image')
INP_PATH = os.path.join('app_data', 'input_image')


# Creating validation and input directories
if os.path.isdir(VAL_PATH) is False:
    os.makedirs(VAL_PATH)
    os.makedirs(INP_PATH)
    # copying 50 random images from anchor folder to validation folder
    for c in random.sample(glob.glob(os.path.join(ANCHOR_PATH, 'anchor_img*')), 50):
        shutil.copy(c, VAL_PATH)


# Verification function
def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join(VAL_PATH)):
        input_img = preprocess(os.path.join(INP_PATH, 'input_image.jpg'))
        validation_img = preprocess(os.path.join('app_data', 'verification_image', image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('app_data', 'verification_image')))
    verified = verification > verification_threshold

    return verified


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = frame[120:370, 200:450, :]

    cv2.imshow('Verification', frame)

    # Press V to verify
    if cv2.waitKey(10) & 0xFF == 118:
        cv2.imwrite(os.path.join('app_data', 'input_image', 'input_image.jpg'), frame)

        verified = verify(final_model, 0.5, 0.5)
        print(verified)

    # Press escape key to escape
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
