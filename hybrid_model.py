import numpy as np
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
import joblib

# Hàm huấn luyện mô hình Hybrid (LSTM + XGBoost)
def hybrid_train(X, y):
    """
    Huấn luyện Hybrid Models (LSTM + XGBoost).
    """
    # Tải mô hình LSTM đã huấn luyện
    lstm_model = load_model('models/lstm_model.h5')
    
    # Dự đoán từ mô hình LSTM
    lstm_features = lstm_model.predict(X)
    
    # Huấn luyện mô hình XGBoost với các đặc trưng từ LSTM
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(lstm_features, y)
    
    # Lưu mô hình XGBoost vào file .pkl
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    
    return lstm_model, xgb_model

# Hàm dự đoán với mô hình Hybrid (LSTM + XGBoost)
def hybrid_predict(lstm_model, xgb_model, last_sequence):
    """
    Dự đoán bằng Hybrid Models (LSTM + XGBoost).
    """
    # Đảm bảo last_sequence có dạng (samples, time_steps, features)
    last_sequence_reshaped = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
    
    # Dự đoán bằng mô hình LSTM
    lstm_features = lstm_model.predict(last_sequence_reshaped)
    
    # Dự đoán bằng mô hình XGBoost với các đặc trưng từ LSTM
    predictions = xgb_model.predict(lstm_features)
    
    # Chuyển kết quả dự đoán thành các số nguyên gần nhất
    return [int(round(num)) for num in predictions[0]]

# Hàm tải mô hình XGBoost từ file .pkl
def load_xgb_model():
    """
    Tải mô hình XGBoost đã lưu.
    """
    return joblib.load('models/xgboost_model.pkl')
