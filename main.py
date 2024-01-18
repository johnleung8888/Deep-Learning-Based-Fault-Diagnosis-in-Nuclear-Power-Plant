import csv
import pandas as pd
import numpy as np
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Read the excel file
data_1 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A1D.xlsx')
data_2 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A2D.xlsx')
data_3 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A3D.xlsx')
data_4 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A4D.xlsx')
data_5 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A5D.xlsx')
data_6 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A6D.xlsx')
data_7 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A7D.xlsx')
data_8 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A8D.xlsx')
data_9 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A9D.xlsx')
data_10 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01A10D.xlsx')
data_11 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D1D.xlsx')
data_12 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D2D.xlsx')
data_13 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D3D.xlsx')
data_14 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D4D.xlsx')
data_15 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D5D.xlsx')
data_16 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D6D.xlsx')
data_17 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D7D.xlsx')
data_18 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D8D.xlsx')
data_19 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D9D.xlsx')
data_20 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS01D10D.xlsx')
data_21 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A1D.xlsx')
data_22 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A2D.xlsx')
data_23 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A3D.xlsx')
data_24 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A4D.xlsx')
data_25 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A5D.xlsx')
data_26 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A6D.xlsx')
data_27 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A7D.xlsx')
data_28 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A8D.xlsx')
data_29 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A9D.xlsx')
data_30 = pd.read_excel(r'C:\Users\JohnLeung\Documents\fypdata\RCS18A10D.xlsx')

# Transform all data into DataFrame
df1 = pd.DataFrame(data_1)
df2 = pd.DataFrame(data_2)
df3 = pd.DataFrame(data_3)
df4 = pd.DataFrame(data_4)
df5 = pd.DataFrame(data_5)
df6 = pd.DataFrame(data_6)
df7 = pd.DataFrame(data_7)
df8 = pd.DataFrame(data_8)
df9 = pd.DataFrame(data_9)
df10 = pd.DataFrame(data_10)
df11 = pd.DataFrame(data_11)
df12 = pd.DataFrame(data_12)
df13 = pd.DataFrame(data_13)
df14 = pd.DataFrame(data_14)
df15 = pd.DataFrame(data_15)
df16 = pd.DataFrame(data_16)
df17 = pd.DataFrame(data_17)
df18 = pd.DataFrame(data_18)
df19 = pd.DataFrame(data_19)
df20 = pd.DataFrame(data_20)
df21 = pd.DataFrame(data_21)
df22 = pd.DataFrame(data_22)
df23 = pd.DataFrame(data_23)
df24 = pd.DataFrame(data_24)
df25 = pd.DataFrame(data_25)
df26 = pd.DataFrame(data_26)
df27 = pd.DataFrame(data_27)
df28 = pd.DataFrame(data_28)
df29 = pd.DataFrame(data_29)
df30 = pd.DataFrame(data_30)

# Normalize all the DataFrame
normalized_df1 = (df1 - df1.min()) / (df1.max() - df1.min())
normalized_df2 = (df2 - df2.min()) / (df2.max() - df3.min())
normalized_df3 = (df3 - df3.min()) / (df3.max() - df3.min())
normalized_df4 = (df4 - df4.min()) / (df4.max() - df4.min())
normalized_df5 = (df5 - df5.min()) / (df5.max() - df5.min())
normalized_df6 = (df6 - df6.min()) / (df6.max() - df6.min())
normalized_df7 = (df7 - df7.min()) / (df7.max() - df7.min())
normalized_df8 = (df8 - df8.min()) / (df8.max() - df8.min())
normalized_df9 = (df9 - df9.min()) / (df9.max() - df9.min())
normalized_df10 = (df10 - df10.min()) / (df10.max() - df10.min())
normalized_df11 = (df11 - df11.min()) / (df11.max() - df11.min())
normalized_df12 = (df12 - df12.min()) / (df12.max() - df12.min())
normalized_df13 = (df13 - df13.min()) / (df13.max() - df13.min())
normalized_df14 = (df14 - df14.min()) / (df14.max() - df14.min())
normalized_df15 = (df15 - df15.min()) / (df15.max() - df15.min())
normalized_df16 = (df16 - df16.min()) / (df16.max() - df16.min())
normalized_df17 = (df17 - df17.min()) / (df17.max() - df17.min())
normalized_df18 = (df18 - df18.min()) / (df18.max() - df18.min())
normalized_df19 = (df19 - df19.min()) / (df19.max() - df19.min())
normalized_df20 = (df20 - df20.min()) / (df20.max() - df20.min())
normalized_df21 = (df21 - df21.min()) / (df21.max() - df21.min())
normalized_df22 = (df22 - df22.min()) / (df22.max() - df22.min())
normalized_df23 = (df23 - df23.min()) / (df23.max() - df23.min())
normalized_df24 = (df24 - df24.min()) / (df24.max() - df24.min())
normalized_df25 = (df25 - df25.min()) / (df25.max() - df25.min())
normalized_df26 = (df26 - df26.min()) / (df26.max() - df26.min())
normalized_df27 = (df27 - df27.min()) / (df27.max() - df27.min())
normalized_df28 = (df28 - df28.min()) / (df28.max() - df28.min())
normalized_df29 = (df29 - df29.min()) / (df29.max() - df29.min())
normalized_df30 = (df30 - df30.min()) / (df30.max() - df30.min())

# Transform all the DataFrame into Numpy Array
normalized_numpy1 = normalized_df1.to_numpy()
normalized_numpy2 = normalized_df2.to_numpy()
normalized_numpy3 = normalized_df3.to_numpy()
normalized_numpy4 = normalized_df4.to_numpy()
normalized_numpy5 = normalized_df5.to_numpy()
normalized_numpy6 = normalized_df6.to_numpy()
normalized_numpy7 = normalized_df7.to_numpy()
normalized_numpy8 = normalized_df8.to_numpy()
normalized_numpy9 = normalized_df9.to_numpy()
normalized_numpy10 = normalized_df10.to_numpy()
normalized_numpy11 = normalized_df11.to_numpy()
normalized_numpy12 = normalized_df12.to_numpy()
normalized_numpy13 = normalized_df13.to_numpy()
normalized_numpy14 = normalized_df14.to_numpy()
normalized_numpy15 = normalized_df15.to_numpy()
normalized_numpy16 = normalized_df16.to_numpy()
normalized_numpy17 = normalized_df17.to_numpy()
normalized_numpy18 = normalized_df18.to_numpy()
normalized_numpy19 = normalized_df19.to_numpy()
normalized_numpy20 = normalized_df20.to_numpy()
normalized_numpy21 = normalized_df21.to_numpy()
normalized_numpy22 = normalized_df22.to_numpy()
normalized_numpy23 = normalized_df23.to_numpy()
normalized_numpy24 = normalized_df24.to_numpy()
normalized_numpy25 = normalized_df25.to_numpy()
normalized_numpy26 = normalized_df26.to_numpy()
normalized_numpy27 = normalized_df27.to_numpy()
normalized_numpy28 = normalized_df28.to_numpy()
normalized_numpy29 = normalized_df29.to_numpy()
normalized_numpy30 = normalized_df30.to_numpy()

# Split the data into training set and validation set
training_array = np.stack(
    [normalized_numpy1, normalized_numpy3, normalized_numpy5, normalized_numpy7, normalized_numpy9,
     normalized_numpy11, normalized_numpy13, normalized_numpy15, normalized_numpy17, normalized_numpy19,
     normalized_numpy21, normalized_numpy23, normalized_numpy25, normalized_numpy27, normalized_numpy29,
     ])

testing_array = np.stack(
    [normalized_numpy2, normalized_numpy4, normalized_numpy6, normalized_numpy12,
     normalized_numpy14, normalized_numpy16, normalized_numpy22, normalized_numpy24, normalized_numpy26])

# Reform the training data in order to fit the neural network
# Setting the labels for the network
train_list = []
test_list = []
counter = 0
num = 0
counter2 = 0
num2 = 0

for i in range(915):
    if counter < 305:
        train_list.append(num)
        counter += 1
    else:
        num += 1
        train_list.append(num)
        counter = 0

for j in range(549):
    if counter2 < 183:
        test_list.append(num2)
        counter2 += 1
    else:
        num += 1
        test_list.append(num2)
        counter2 = 0

training_labels = np.asfarray(train_list, float)
testing_labels = np.asfarray(test_list, float)

training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=3)
testing_labels = tf.keras.utils.to_categorical(testing_labels, num_classes=3)

training_array = training_array.reshape(915, 11, 8)
testing_array = testing_array.reshape(549, 11, 8)

training_array = tf.expand_dims(training_array, axis=3)
testing_array = tf.expand_dims(testing_array, axis=3)

# CNN model structure
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(11, 8, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='tanh'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Print the model summary
model.summary()

# Model compiler
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=5e-6),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Fit the model and print out the result
history = model.fit(x=training_array, y=training_labels, batch_size=16, epochs=100,
                    validation_data=(testing_array, testing_labels))

plt.subplot(211)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()
