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
error_log_file = "Statistics/error_log.txt"

# Function to extract EXIF data with error handling
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
    except (AttributeError, KeyError, TypeError, IndexError) as e:
        print(f"Corrupt EXIF data or issue with image: {img_path}")
        return None
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return None

try:
    ai_gen_path = "C:/Users/Tanmay Somani/OneDrive/Desktop/Programming/!Projects/FACT/Image/Dataset/AiArtData/AiArtData"
    real_art_path = "C:/Users/Tanmay Somani/OneDrive/Desktop/Programming/!Projects/FACT/Image/Dataset/RealArt/RealArt"
    categories = ['Real', 'AIGenerated']
    img_size = 48
    training_data = []

    for category in categories:
        path = ai_gen_path if category == 'AIGenerated' else real_art_path
        classes = categories.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # Extract EXIF data
            exif_data = extract_exif_data(img_path)
            exif_feature = 1 if exif_data else 0  # 1 if EXIF data exists, 0 if not
            
            # Read and process the image
            img_array = cv.imread(img_path)
            new_array = cv.resize(img_array, (img_size, img_size))
            new_array = new_array / 255

            # Append the image and EXIF feature to the training data
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
    exif_features = np.array(exif_features)

    with open("X_train.pickle", "wb") as pickle_out:
        pickle.dump(X_train, pickle_out, protocol=4)

    with open("y_train.pickle", "wb") as pickle_out:
        pickle.dump(y_train, pickle_out, protocol=4)

    with open("exif_features.pickle", "wb") as pickle_out:
        pickle.dump(exif_features, pickle_out, protocol=4)

    with open("X_train.pickle", "rb") as pickle_in:
        X_train = pickle.load(pickle_in)

    with open("y_train.pickle", "rb") as pickle_in:
        y_train = pickle.load(pickle_in)

    with open("exif_features.pickle", "rb") as pickle_in:
        exif_features = pickle.load(pickle_in)

    print("Loaded X_train shape:", X_train.shape)
    print("Loaded y_train shape:", y_train.shape)
    print("Loaded exif_features shape:", exif_features.shape)
    
    import tensorflow as tf
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(48,48,3)),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(128,(3,3), activation='relu'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(256,(3,3), activation='relu'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=15)

    csv_file = "Statistics/execution_times.csv"
    if not os.path.exists("Statistics"):
        os.makedirs("Statistics")

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

    print(f"Time of program execution: {execution_time:.2f} seconds")

except Exception:
    if not os.path.exists("Statistics"):
        os.makedirs("Statistics")
    with open(error_log_file, mode='a') as log_file:
        log_file.write("-----\n")
        log_file.write(f"Error occurred at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(traceback.format_exc())
        log_file.write("-----\n")
    print("An error occurred. Please check the error log file for details.")
