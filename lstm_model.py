from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape):
    """
    Xây dựng mô hình LSTM.
    :param input_shape: Hình dạng đầu vào (samples, time_steps, features)
    :return: Mô hình LSTM
    """
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(input_shape[2], activation='linear')  # Dự đoán theo số lượng đặc trưng
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_lstm_model(X, y, epochs=50, batch_size=16):
    """
    Huấn luyện mô hình LSTM.
    :param X: Dữ liệu đầu vào
    :param y: Nhãn (giá trị cần dự đoán)
    :param epochs: Số vòng lặp
    :param batch_size: Kích thước batch
    :return: Mô hình đã huấn luyện
    """
    model = build_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save('models/lstm_model.h5')
    return model
