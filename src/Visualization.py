# Image Processing
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Text Processing
from textwrap import wrap

# MLflow
import mlflow
import mlflow.tensorflow


def readImage(path, img_size=224):
    img = load_img(
        path,
        color_mode='rgb',
        target_size=(img_size, img_size)
    )

    img = img_to_array(img)
    img = img / 255.0

    return img


def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)

    plt.figure(figsize=(20, 20))

    n = 0

    for i in range(min(15, len(temp_df))):
        n += 1

        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)

        image = readImage(
            f"../input/flickr8k/Images/{temp_df.image[i]}"
        )

        plt.imshow(image)

        plt.title("\n".join(wrap(temp_df.caption[i], 20)))

        plt.axis("off")

    plt.show()


display_images(data.sample(15))

# Set MLflow experiment
mlflow.set_experiment("Flickr8K_Image_Captioning")

# Start MLflow run
with mlflow.start_run(run_name="Image_Captioning_Experiment"):

    # Log parameters
    mlflow.log_param("image_size", 224)
    mlflow.log_param("dataset", "Flickr8K")
    mlflow.log_param("color_mode", "RGB")
    mlflow.log_param("normalization", "pixel_value_divided_by_255")

    # Example metrics (replace with actual training metrics)
    # mlflow.log_metric("train_loss", train_loss)
    # mlflow.log_metric("validation_loss", val_loss)

    print("MLflow tracking initialized successfully!")