from PBL import L1Dist
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Kiểm tra xem mô hình đã tồn tại chưa
MODEL_PATH = 'siamesemodelv2.h5'
if os.path.exists(MODEL_PATH):
    # Nếu đã có mô hình, tải mô hình thay vì huấn luyện lại
    print("Loading pre-trained Siamese model...")
    siamese_model = load_model(MODEL_PATH, custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
else:
    # Nếu chưa có mô hình, tiến hành huấn luyện mô hình
    print("Training Siamese model...")
    siamese_model = make_siamese_model()
    train(train_data, EPOCHS)
    siamese_model.save(MODEL_PATH)
    print("Model saved to disk.")
