import time
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import traceback
import random
import pickle
import cv2 as cv
from PIL import Image
from PIL.ExifTags import TAGS

start_time = time.time()
error_log_file = "Image/Statistics/error_log.txt"

def extract_exif_data(img_path):
    try:
        img = Image.open(img_path)
        exif_data = img._getexif()
        if exif_data is None:
            return None
        extracted_info = {}
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            extracted_info[tag_name] = value
        return extracted_info
    except (AttributeError, KeyError, TypeError, IndexError):
        return None
    except Exception:
        return None

try:
    ai_gen_path = "C:/Users/Tanmay Somani/OneDrive/Desktop/Programming/!Projects/FACT/Image/Dataset/AiArtData/AiArtData"
    real_art_path = "C:/Users/Tanmay Somani/OneDrive/Desktop/Programming/!Projects/FACT/Image/Dataset/RealArt/RealArt"
    categories = ['Real', 'AIGenerated']
    img_size = 48
    training_data = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for category in categories:
        path = ai_gen_path if category == 'AIGenerated' else real_art_path
        classes = categories.index(category)
        for img in os.listdir(path):
            if not img.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(path, img)
            exif_data = extract_exif_data(img_path)
            exif_feature = 1 if exif_data else 0
            img_array = cv.imread(img_path)
            if img_array is None:
                continue
            new_array = cv.resize(img_array, (img_size, img_size)) / 255.0
            training_data.append([new_array, classes, exif_feature])

    random.shuffle(training_data)
    X_train = []
    y_train = []
    exif_features = []

    for features, label, exif in training_data:
        X_train.append(features)
        y_train.append(label)
        exif_features.append(exif)

    X_train = np.array(X_train).reshape(-1, img_size, img_size, 3)
    y_train = np.array(y_train)
    exif_features = np.array(exif_features).reshape(-1, 1)

    with open("Image\Train\X_train.pickle", "wb") as pickle_out:
        pickle.dump(X_train, pickle_out, protocol=4)
    with open("Image\Train\y_train.pickle", "wb") as pickle_out:
        pickle.dump(y_train, pickle_out, protocol=4)
    with open("Image\Train\exif_features.pickle", "wb") as pickle_out:
        pickle.dump(exif_features, pickle_out, protocol=4)
    with open("Image\Train\X_train.pickle", "rb") as pickle_in:
        X_train = pickle.load(pickle_in)
    with open("Image\Train\y_train.pickle", "rb") as pickle_in:
        y_train = pickle.load(pickle_in)
    with open("Image\Train\exif_features.pickle", "rb") as pickle_in:
        exif_features = pickle.load(pickle_in)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, Concatenate

    image_input = Input(shape=(48, 48, 3), name='image_input')
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    exif_input = Input(shape=(1,), name='exif_input')
    y = Dense(16, activation='relu')(exif_input)
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dense(64, activation='relu')(z)
    output = Dense(1, activation='sigmoid')(z)
    model = keras.Model(inputs=[image_input, exif_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([X_train, exif_features], y_train, epochs=15, batch_size=32)

    testing_data = []
    for category in categories:
        path = ai_gen_path if category == 'AIGenerated' else real_art_path
        classes = categories.index(category)
        for img in os.listdir(path):
            if not img.lower().endswith(valid_extensions):
                continue
            img_path = os.path.join(path, img)
            exif_data = extract_exif_data(img_path)
            exif_feature = 1 if exif_data else 0
            img_array = cv.imread(img_path)
            if img_array is None:
                continue
            new_array = cv.resize(img_array, (img_size, img_size)) / 255.0
            testing_data.append([new_array, classes, exif_feature])

    random.shuffle(testing_data)
    X_test = []
    y_test = []
    exif_features_test = []
    for features, label, exif in testing_data:
        X_test.append(features)
        y_test.append(label)
        exif_features_test.append(exif)
    X_test = np.array(X_test).reshape(-1, img_size, img_size, 3)
    y_test = np.array(y_test)
    exif_features_test = np.array(exif_features_test).reshape(-1, 1)
    loss, accuracy = model.evaluate([X_test, exif_features_test], y_test)
    model.save("Image/saved_model/my_ai_art_classifier")
    from tensorflow.keras.models import load_model
    model = load_model("Image/saved_model/my_ai_art_classifier")
    loss, accuracy = model.evaluate([X_test, exif_features_test], y_test)
    y_pred = model.predict([X_test, exif_features_test])

    csv_file = "Image/Statistics/execution_times.csv"
    if not os.path.exists("Image/Statistics"):
        os.makedirs("Image/Statistics")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Execution Run", "Execution Time (seconds)", "Total Accumulated Time (seconds)"])
    total_time = 0
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                total_time += float(row[1])
    execution_time = time.time() - start_time
    total_time += execution_time
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        run_count = sum(1 for _ in open(csv_file)) - 1
        writer.writerow([run_count, f"{execution_time:.2f}", f"{total_time:.2f}"])
except Exception:
    if not os.path.exists("Image/Statistics"):
        os.makedirs("Image/Statistics")
    with open(error_log_file, mode='a') as log_file:
        log_file.write("-----\n")
        log_file.write(f"Error occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(traceback.format_exc())
        log_file.write("-----\n")