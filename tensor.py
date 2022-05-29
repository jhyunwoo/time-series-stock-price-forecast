import tensorflow as tf

import pandas as pd

data = pd.read_csv('gpascore.csv')
print(data)


print(data.isnull().sum())
data = data.dropna()
print(data.isnull().sum())
exit()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='softmax'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit()