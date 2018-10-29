from keras.models import load_model, Model
import numpy as np
from keras.preprocessing import image
import os

classifier = load_model(os.path.join(os.getcwd(), 'model/my_model.h5'))

img_pred = image.load_img(os.path.join(os.getcwd(), 'dataset/test/predict5.jpg', target_size = (64, 64))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = classifier.predict(img_pred)
if rslt[0][0] == 1:
    print("crack")
else:
    print("NO")
