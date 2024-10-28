import cv2
import os
import random
import numpy as np
import uuid
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))

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
plt.imshow(frame[120:120+250,200:200+250, :])

anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(3000)

dir_test = anchor.as_numpy_iterator()

print(dir_test.next())

# # Chỉnh lại size ảnh


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

img = preprocess('data\\anchor\\p2.jpg')
img.numpy().max() 

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

exampple = samples.next()

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)
res = preprocess_twin(*exampple)

plt.imshow(res[1])

data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

inp = Input(shape=(100,100,3), name='input_image')


c1 = Conv2D(64, (10,10), activation='relu')(inp)

m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

c2 = Conv2D(128, (7,7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

c3 = Conv2D(128, (4,4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

c4 = Conv2D(256, (4,4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)


mod = Model(inputs=[inp], outputs=[d1], name='embedding')

mod.summary()

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

embedding = make_embedding()

embedding.summary()

class L1Dist(Layer):   
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
l1 = L1Dist()

input_image = Input(name='input_img', shape=(100,100,3))
validation_image = Input(name='validation_img', shape=(100,100,3))

inp_embedding = embedding(input_image)
val_embedding = embedding(validation_image)

siamese_layer = L1Dist()

distances = siamese_layer(inp_embedding, val_embedding)


classifier = Dense(1, activation='sigmoid')(distances)

siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_network.summary()

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
siamese_model = make_siamese_model()
siamese_model.summary()
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

test_batch = train_data.as_numpy_iterator()

batch_1 = test_batch.next()

X = batch_1[:2]
y = batch_1[2]
y
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

EPOCHS = 50
train(train_data, EPOCHS)
from tensorflow.keras.metrics import Precision, Recall

test_input, test_val, y_true = test_data.as_numpy_iterator().next()
y_hat = siamese_model.predict([test_input, test_val])
[1 if prediction > 0.5 else 0 for prediction in y_hat ]

# Tạo một đối tượng metric Recall
m = Recall()

# Tính toán giá trị recall
m.update_state(y_true, y_hat)

# Trả về kết quả Recall
m.result().numpy()
r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())
siamese_model.save('siamesemodelv2.h5')
# Reload model 
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
# Dự đoán bằng model
siamese_model.predict([test_input, test_val])


siamese_model.summary()

os.listdir(os.path.join('application_data', 'verification_images'))

os.path.join('application_data', 'input_image', 'input_image.jpg')

for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    print(validation_img)

cap = cv2.VideoCapture(0)  # Mở kết nối đến webcam
while cap.isOpened():  # Khi webcam mở
    ret, frame = cap.read()  # Đọc khung hình từ webcam
    frame = frame[120:120+250, 200:200+250, :]  # Cắt khung hình thành kích thước 250x250px
    
    cv2.imshow('Verification', frame)  # Hiển thị khung hình trong cửa sổ 'Verification'
    
    # Kích hoạt xác minh
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Lưu hình ảnh đầu vào vào thư mục application_data/input_image
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        
        # Thực hiện xác minh và nhận diện danh tính
        results, verified_identity = verify(siamese_model, 0.5, 0.5)
        if verified_identity:
            print(f"Nhận diện thành công: {verified_identity}")
        else:
            print("Không nhận diện được khuôn mặt.")
    
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Nếu nhấn 'q', thoát khỏi vòng lặp
        break

cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)  # Mở kết nối đến webcam
while cap.isOpened():  # Khi webcam mở
    ret, frame = cap.read()  # Đọc khung hình từ webcam
    frame = frame[120:120+250, 200:200+250, :]  # Cắt khung hình thành kích thước 250x250px
    
    cv2.imshow('Verification', frame)  # Hiển thị khung hình trong cửa sổ 'Verification'
    
    # Kích hoạt xác minh
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Lưu hình ảnh đầu vào vào thư mục application_data/input_image
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        
        # Thực hiện xác minh và nhận diện danh tính
        results, verified_identity = verify(siamese_model, 0.5, 0.5)
        if verified_identity:
            print(f"Nhận diện thành công: {verified_identity}")
        else:
            print("Không nhận diện được khuôn mặt.")
    
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Nếu nhấn 'q', thoát khỏi vòng lặp
        break

cap.release()
cv2.destroyAllWindows()
