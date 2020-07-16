# %%
import tensorflow.keras as keras
import tensorflow as tf

# %%
# load Mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# %%
#print(x_train[0])

# %%
import matplotlib.pyplot as plt

# %%
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

# %%
print(y_train[0])

# %%
# Normalise data so instead of range 0 to 255 it's 0 to 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# %%
#print(x_train[0])

# %%
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

# %%
# build model
model = tf.keras.models.Sequential()
# add flat layer (input layer)
model.add(tf.keras.layers.Flatten())
# add hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add a second identical layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# %%
# train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# fit
model.fit(x_train, y_train, epochs=3)

# %%
# check loss and accuracy on test set
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

# %%
# save model
model.save('epic_num_reader.model')

# %%
# load model
new_model = tf.keras.models.load_model('epic_num_reader.model')

# %%
predictions = new_model.predict(x_test)

# %%
#print(predictions)

# %%
import numpy as np

print(np.argmax(predictions[0]))

# %%
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

# %%
