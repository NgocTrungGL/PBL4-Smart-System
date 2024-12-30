#!/usr/bin/env python
# coding: utf-8

# In[73]:


import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[74]:


img = cv.imread("./dataset/sardor_abdirayimov/5.png") #ảnh là BGR


# In[75]:


img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img) #RGB


# In[76]:


from mtcnn.mtcnn import MTCNN

detector = MTCNN()
results = detector.detect_faces(img)


# In[77]:


results


# In[78]:


x,y,w,h = results[0]['box'] #'box': [469, 316, 246, 333],


# In[79]:


img = cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 30)
plt.imshow(img)


# In[80]:


my_face = img[y:y+h, x:x+w]
#Facenet takes as input 160x160 
my_face = cv.resize(my_face, (160,160))
plt.imshow(my_face)


# In[81]:


my_face


# In[82]:


class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()
    

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr
    

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        
        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')


# In[83]:


faceloading = FACELOADING("./dataset")
X, Y = faceloading.load_classes()


# In[84]:


plt.figure(figsize=(16,12))
for num,image in enumerate(X):
    ncols = 3
    nrows = len(Y)//ncols + 1
    plt.subplot(nrows,ncols,num+1)
    plt.imshow(image)
    plt.axis('off')


# In[85]:


from keras_facenet import FaceNet
embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0) 
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)


# In[86]:


EMBEDDED_X = []

for img in X:
    EMBEDDED_X.append(get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)


# In[87]:


np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)


# # SVM MODEL

# In[88]:


from sklearn.preprocessing import LabelEncoder
# Nếu Y chứa các nhãn ban đầu như ['person1', 'person2', 'person3'], sau mã hóa sẽ thành [0, 1, 2].
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)


# In[89]:


plt.plot(EMBEDDED_X[0]) 
plt.ylabel(Y[0])


# In[90]:


# break


# In[91]:


Y


# In[92]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)


# In[93]:


from sklearn.svm import SVC
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)


# In[94]:


ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)


# In[95]:


from sklearn.metrics import accuracy_score

accuracy_score(Y_train, ypreds_train)


# In[96]:


accuracy_score(Y_test, ypreds_test)


# In[97]:


t_im = cv.imread("./sardor_test.jpg")
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x,y,w,h = detector.detect_faces(t_im)[0]['box']


# In[98]:


t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160,160))
test_im = get_embedding(t_im)


# In[99]:


test_im = [test_im]
ypreds = model.predict(test_im)


# In[100]:


ypreds


# In[101]:


encoder.inverse_transform(ypreds)


# In[102]:


import pickle
#save the model
with open('svm_model_160x160.pkl','wb') as f:
    pickle.dump(model,f)

