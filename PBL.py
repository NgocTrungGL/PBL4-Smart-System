#!/usr/bin/env python
# coding: utf-8

# # Tải các thư viện

# In[3]:


import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


# In[4]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf


# # Set GPU

# In[5]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# # Folder path

# In[6]:


POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


# # Thu dữ liệu vào folder Positive và anchor

# In[7]:


import uuid


# In[8]:


os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))


# In[9]:


# Thiết lập kết nối với webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    # Cắt khung hình thành kích thước 250x250px
    frame = frame[120:120+250, 200:200+250, :]
    
    # Thu thập hình ảnh anchor 
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Tạo đường dẫn tệp tin duy nhất 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Ghi hình ảnh anchor ra tệp
        cv2.imwrite(imgname, frame)
    
    # Thu thập hình ảnh positive
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Tạo đường dẫn tệp tin duy nhất 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Ghi hình ảnh positive ra tệp
        cv2.imwrite(imgname, frame)
    
    # Hiện hình ảnh lên màn hình
    cv2.imshow('Image Collection', frame)
    
    # Thoát chương trình một cách an toàn
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Giải phóng webcam
cap.release()
# Đóng khung hình hiển thị hình ảnh
cv2.destroyAllWindows()


# In[10]:


plt.imshow(frame[120:120+250,200:200+250, :])


# # Tạo biến chứa ảnh

# In[11]:


anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)


# In[12]:


dir_test = anchor.as_numpy_iterator()


# In[13]:


print(dir_test.next())


# # Chỉnh lại size ảnh

# In[14]:


def preprocess(file_path):
    # Đọc hình ảnh từ đường dẫn tệp
    byte_img = tf.io.read_file(file_path)
    # Giải mã hình ảnh từ dạng JPEG
    img = tf.io.decode_jpeg(byte_img)
    
    # Các bước tiền xử lý - thay đổi kích thước hình ảnh về 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Chuẩn hóa hình ảnh để giá trị nằm trong khoảng từ 0 đến 1
    img = img / 255.0

    # Trả về hình ảnh đã được xử lý
    return img


# In[15]:


img = preprocess('data\\anchor\\p2.jpg')


# In[16]:


img.numpy().max() 


# # Tạo Dataset

# In[17]:


positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


# In[18]:


samples = data.as_numpy_iterator()


# In[19]:


exampple = samples.next()


# In[20]:


exampple


# In[21]:


import pandas as pd
import tensorflow as tf

image_pairs = []
labels = []

# Duyệt qua các phần tử trong tập dữ liệu và thêm vào danh sách
for anchor_img, positive_or_negative_img, label in data:
    # Giả sử chúng ta chuyển đổi các tensor hình ảnh thành numpy array
    anchor_img = anchor_img.numpy()  # Chuyển đổi tensor thành numpy array
    positive_or_negative_img = positive_or_negative_img.numpy()  # Chuyển đổi tensor thành numpy array
    
    image_pairs.append((anchor_img, positive_or_negative_img))
    labels.append(label.numpy())

# Tạo DataFrame từ các danh sách
df = pd.DataFrame(image_pairs, columns=['Anchor', 'Image'])
df['Label'] = labels

# Hiển thị DataFrame
display(df)


# # Xây dựng phân chia giữa tập huấn luyện và tập kiểm tra

# In[22]:


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# In[23]:


res = preprocess_twin(*exampple)


# In[24]:


plt.imshow(res[1])


# In[25]:


res[2]


# In[26]:


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)


# In[27]:


train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


# In[28]:


test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# # Tạo lớp embedding

# In[29]:


inp = Input(shape=(100,100,3), name='input_image')


# In[30]:


c1 = Conv2D(64, (10,10), activation='relu')(inp)


# In[31]:


m1 = MaxPooling2D(64, (2,2), padding='same')(c1)


# In[32]:


c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)


# In[33]:


c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)


# In[34]:


c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)


# In[35]:


mod = Model(inputs=[inp], outputs=[d1], name='embedding')


# In[36]:


mod.summary()


# In[37]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def make_embedding(): 
    inp = Input(shape=(100, 100, 3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=inp, outputs=d1, name='embedding')


# In[38]:


embedding = make_embedding()


# In[39]:


embedding.summary()


# In[40]:


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# In[41]:


l1 = L1Dist()


# # Tạo Module (siamese)

# In[42]:


input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))


# In[43]:


inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)


# In[44]:


siamese_layer = L1Dist()


# In[45]:


distances = siamese_layer(inp_embedding, val_embedding)


# In[46]:


classifier = Dense(1, activation='sigmoid')(distances)


# In[47]:


classifier


# In[48]:


siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[49]:


siamese_network.summary()


# In[50]:


def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# In[51]:


siamese_model = make_siamese_model()


# In[52]:


siamese_model.summary()


# # Training

# In[53]:


binary_cross_loss = tf.losses.BinaryCrossentropy()


# In[54]:


opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


# In[55]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# In[56]:


test_batch = train_data.as_numpy_iterator()


# In[57]:


batch_1 = test_batch.next()


# In[58]:


X = batch_1[:2]


# In[59]:


y = batch_1[2]


# In[60]:


y


# In[61]:


get_ipython().run_line_magic('pinfo2', 'tf.losses.BinaryCrossentropy')


# In[62]:


@tf.function
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss


# In[63]:


from tensorflow.keras.metrics import Precision, Recall


# In[64]:


def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


# In[65]:


EPOCHS = 50


# In[66]:


# train(train_data, EPOCHS)


# In[67]:


from tensorflow.keras.metrics import Precision, Recall


# In[68]:


test_input, test_val, y_true = test_data.as_numpy_iterator().next()


# In[69]:


y_hat = siamese_model.predict([test_input, test_val])


# In[70]:


[1 if prediction > 0.5 else 0 for prediction in y_hat ]


# In[71]:


y_true


# In[72]:


# Tạo một đối tượng metric Recall
m = Recall()

# Tính toán giá trị recall
m.update_state(y_true, y_hat)

# Trả về kết quả Recall
m.result().numpy()


# In[73]:


r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())


# In[74]:


# Đặt kích thước biểu đồ 
plt.figure(figsize=(10,8))

# Đặt biểu đồ con đầu tiên
plt.subplot(1, 2, 1)
plt.imshow(test_input[0])

# Đặt biểu đồ con thứ hai
plt.subplot(1, 2, 2)
plt.imshow(test_val[0])

# Hiển thị biểu đồ
plt.show()


# In[75]:


siamese_model.save('siamesemodelv2.h5')


# In[76]:


L1Dist


# In[77]:


# Reload model 
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[78]:


# Dự đoán bằng model
siamese_model.predict([test_input, test_val])


# In[79]:


siamese_model.summary()


# In[80]:


os.listdir(os.path.join('application_data', 'verification_images'))


# In[81]:


os.path.join('application_data', 'input_image', 'input_image.jpg')


# In[82]:


for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    print(validation_img)


# In[83]:


def verify(model, detection_threshold, verification_threshold):
    # Tạo mảng kết quả
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Dự đoán
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    # Ngưỡng phát hiện: Chỉ số trên đó một dự đoán được coi là dương
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Ngưỡng xác minh: Tỷ lệ dự đoán dương / tổng số mẫu dương
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return results, verified


# In[84]:


cap = cv2.VideoCapture(0)  # Mở kết nối đến webcam
while cap.isOpened():  # Khi webcam mở
    ret, frame = cap.read()  # Đọc khung hình từ webcam
    frame = frame[120:120+250, 200:200+250, :]  # Cắt khung hình thành kích thước 250x250px
    
    cv2.imshow('Verification', frame)  # Hiển thị khung hình trong cửa sổ 'Verification'
    
    # Kích hoạt xác minh
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Lưu hình ảnh đầu vào vào thư mục application_data/input_image
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Thực hiện xác minh
        results, verified = verify(siamese_model, 0.5, 0.5)
        print(verified)  # In kết quả xác minh
    
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Nếu nhấn 'q', thoát khỏi vòng lặp
        break

cap.release()  # Giải phóng webcam
cv2.destroyAllWindows()  # Đóng tất cả cửa sổ


# 

# In[85]:


np.sum(np.squeeze(results) > 0.9)


# In[86]:


results


# In[ ]:




