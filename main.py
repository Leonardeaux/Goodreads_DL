import matplotlib.pyplot as plt
import os
import re
import string
import tensorflow as tf
import pandas as pd

from utils import predicted_test_data_to_result_csv
from tensorflow.keras import layers
from tensorflow.keras import losses

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print(os.environ['LD_LIBRARY_PATH'])
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

df_train = pd.read_csv("data/base/goodreads_train.csv", sep=",")
df_train.head()

index = df_train[(df_train['rating'] == 0)].index
df_train.drop(index, inplace=True)
df_train.reset_index(inplace=True, drop=True)
df_train.head()

target = df_train.pop('rating')

target = target - 1

target.head()

features = df_train["review_text"]

features.head()

raw_train_ds = tf.data.Dataset.from_tensor_slices((features, target))
raw_train_ds = raw_train_ds.batch(32)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_spoilers = tf.strings.regex_replace(lowercase, '\*\* spoiler alert \*\*', ' ')
    return tf.strings.regex_replace(stripped_spoilers,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


max_features = 10000
sequence_length = 100

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = raw_train_ds.map(vectorize_text)

embedding_dim = 200

model = tf.keras.Sequential()

model.add(layers.Embedding(max_features + 1, 50, input_length=sequence_length))

model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))

model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.summary()

model.compile(loss=losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

epochs = 20
history = model.fit(
    train_ds,
    epochs=epochs)
