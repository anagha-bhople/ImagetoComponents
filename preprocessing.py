import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization
seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

# Desired image dimensions
IMAGE_SIZE = (512, 512)

# Vocabulary size
VOCAB_SIZE = 500

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 32
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

def load_captions_data(PATH):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """
    import os
    # for file as os.listdir(PATH+'TEXT_LABELS'):
    #     caption_data = caption_file.readlines()
    caption_mapping = {}
    text_data = []
        # images_to_skip = set()
    for filename in os.listdir(PATH+'/TEXT_LABELS/'):
    
        with open(PATH+'/TEXT_LABELS/'+filename) as file:
            file_data = file.readlines()
            line = " ".join(file_data)
            line = line.replace("\n"," <NEWLINE>")
            line = line.replace("{","<BRACES>")
            line = line.replace("}","<BRACEE>")
            # words = line.split(' ')
            # Image name and captions are separated using a tab
            img_name, caption = filename, line

            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            # img_name = img_name.split("#")[0]
            img_name = os.path.join(PATH+'/IMAGES/', img_name.strip()[:-3]+'png')

            # We will remove caption that are either too short to too long
            # tokens = words

            # if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
            #     images_to_skip.add(img_name)
            #     continue

            # if img_name.endswith("jpg") and img_name not in images_to_skip:
            #     # We will add a start and an end token to each caption
            caption = "<start> " + caption.strip() + " <end>"
            text_data.append(caption)

            if img_name in caption_mapping:
                caption_mapping[img_name].append(caption)
            else:
                caption_mapping[img_name] = [caption]

        # for img_name in images_to_skip:
        #   if img_name in caption_mapping:
        #       del caption_mapping[img_name]

    return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data




def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img



strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    # return lowercase
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")




