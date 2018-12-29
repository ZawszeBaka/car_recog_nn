import pickle
import cv2
import scipy
from PIL import Image
from scipy import ndimage
from tf_utils import *
import matplotlib.pyplot as plt

def load_parameters(file_name):
    with open('model'+'/'+file_name,'rb') as f:
        return pickle.load(f)

def predict_process(img_path,parameters,size=(64,64)):
    img = cv2.imread(img_path)
    if img is None:
        return
    if img[0,0].shape==(3,):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)

    print('[DEBUG] shape', gray.shape)
    gray = gray.reshape(gray.shape[0]*gray.shape[1],1)
    info, prediction = predict(gray,parameters)
    print('[INFO] Prediction : ', prediction)
    plt.imshow(img)
    info /= np.sum(info)
    info = 1/(1+np.exp(-info))
    if prediction == 1:
        plt.title('[INFO] It is a car ' + str(info[prediction,0][0]) + '%')
    else:
        plt.title('[INFO] It is not a car ' + str(info[prediction,0][0]) + '%')
    plt.show()


if __name__ == '__main__':

    [costs, parameters] = load_parameters('car_model.pickle')
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 5)')
    plt.title('Learning rate = '+str(0.0001))
    plt.show()
    predict_process('images/car-1.jpg', parameters,size=(64,64))
    predict_process('images/non-car-1.jpg', parameters,size=(64,64))
