import cv2
import os
import random
import numpy as np
import uuid
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall

# Thiết lập GPU (nếu có)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Đường dẫn đến thư mục dữ liệu
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# Kết nối với webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
    
    # Cắt khung hình thành kích thước 250x250px
    frame = frame[120:120+250, 200:200+250, :]
    
    # Thu thập hình ảnh anchor
    if cv2.waitKey(1) & 0XFF == ord('a'):
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Thu thập hình ảnh positive cho người 1
    if cv2.waitKey(1) & 0XFF == ord('p'):
        imgname = os.path.join(POS_PATH, 'person1_{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    
    # Thu thập hình ảnh positive cho người 2
    if cv2.waitKey(1) & 0XFF == ord('o'):
        imgname = os.path.join(POS_PATH, 'person2_{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    # Hiện hình ảnh lên màn hình
    cv2.imshow('Image Collection', frame)
    
    # Thoát chương trình một cách an toàn
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Giải phóng webcam
cap.release()
cv2.destroyAllWindows()

# Chuẩn bị dữ liệu cho training
anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(3000)
positive_person1 = tf.data.Dataset.list_files(POS_PATH + '/person1_*.jpg').take(3000)
positive_person2 = tf.data.Dataset.list_files(POS_PATH + '/person2_*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(3000)

# Tiến hành tổ chức dữ liệu cho mô hình
positives_person1 = tf.data.Dataset.zip((anchor, positive_person1, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
positives_person2 = tf.data.Dataset.zip((anchor, positive_person2, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

# Kết hợp tất cả dữ liệu
data = positives_person1.concatenate(positives_person2).concatenate(negatives)

# Hàm tiền xử lý ảnh
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

# Tiến hành xử lý cặp ảnh
def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

data = data.map(preprocess_twin)
data = data.cache().shuffle(buffer_size=10000)

# Chia dữ liệu thành train và test
train_data = data.take(round(len(data) * .7)).batch(16).prefetch(8)
test_data = data.skip(round(len(data) * .7)).take(round(len(data) * .3)).batch(16).prefetch(8)

# Xây dựng mô hình Siamese
def make_embedding(): 
    inp = Input(shape=(100, 100, 3), name='input_image')
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(2, 2))(c1)
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(pool_size=(2, 2))(c2)
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(pool_size=(2, 2))(c3)
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return Model(inputs=inp, outputs=d1, name='embedding')

embedding = make_embedding()

# Lớp L1 Distance
class L1Dist(Layer):   
    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Xây dựng mô hình Siamese
def make_siamese_model(): 
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    
    siamese_layer = L1Dist()
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()

# Huấn luyện mô hình
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:     
        X = batch[:2]
        y = batch[2]
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss

def train(data, EPOCHS):
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        for idx, batch in enumerate(data):
            loss = train_step(batch)
        print(f'Loss: {loss.numpy()}')

EPOCHS = 50
train(train_data, EPOCHS)

# Lưu mô hình
siamese_model.save('siamesemodelv2.h5')
