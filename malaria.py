import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten
import matplotlib.pyplot as plt

#path = 'C101P62ThinF_IMG_20150918_151507_cell_21.png'
class Cell:
    def __init__(self,path, model_path='malaria_cnn.h5'):
        self.path = path
        self.model_path = model_path

    def ImgRead(self):
        img = cv2.imread(self.path)
        img = Image.fromarray(img,'RGB')
        img = img.resize((64,64))
        img_array = np.array(img)
        return img_array

    def detect(self):
        img_processed = self.ImgRead()
        img_processed = img_processed.reshape(1,64,64,3)
        model = load_model(self.model_path)
        y_hat = model.predict(img_processed)
        y_con = model.predict_proba(img_processed)
        prediction = ['Parasitized' if x<0.5 else 'Uninfected' for x in y_hat]
        confidence = [int(y_con[0][0]*100) if prediction[0] is 'Uninfected' else int((1-y_con[0][0])*100)]
        return prediction[0],confidence[0]



#a = ImgRead('C116P77ThinF_IMG_20150930_171809_cell_80.png')
#plt.imshow(a)
#plt.show()
#model = load_model('malaria_cnn.h5')
#print(model.summary())

#y_hat = model.predict(a.reshape(1,64,64,3))
#prediction = ['Parasitized' if x<0.5 else 'Uninfected' for x in y_hat]
#print('Status:',prediction[0])
