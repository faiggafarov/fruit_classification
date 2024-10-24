import os
from keras.preprocessing import image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive) 
import matplotlib.pyplot as plt
import keras

model_predict = keras.models.load_model('model.h5')
model_predict.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

image_name = []
image_conf = []
predict_result = []

# Assuming you've already uploaded the image 'alma.jpg'

path = 'alma.jpg'
img = image.load_img(path, color_mode="rgb", target_size=(224, 224), interpolation="lanczos")
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255

images = np.vstack([img])
classes = model_predict.predict(images, batch_size=10)
label = np.where(classes[0] > 0.5, 1, 0)

if label == 0:
    predict_result.append('Fresh Fruit')
    image_conf.append(1.0 - classes[0])
else:
    predict_result.append('Rotten Fruit')
    image_conf.append(classes[0])

plt.figure(figsize=(8, 8))
plt.imshow(image.load_img(path, color_mode="rgb", target_size=(150, 150), interpolation="bicubic"))
title = f"predict: {predict_result[0]} ({round(float(image_conf[0])*100, 2)}%)"

if predict_result[0] == 'Fresh Fruit':
    plt.title(title, fontsize=18, color='teal')
else:
    plt.title(title, fontsize=18, color='maroon')

plt.axis('off')
plt.show()
plt.savefig('fruit_prediction.png')
os.remove(path)
