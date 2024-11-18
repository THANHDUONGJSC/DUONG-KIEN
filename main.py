import tkinter as tk
from tkinter import filedialog, Toplevel  # Thêm Toplevel vào phần import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from database import LotteryDatabase
import os

class LotteryPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NXTD - Dự đoán KQXS")
        self.data = None
        self.save_folder = None  # Thư mục lưu trữ dữ liệu
        
        # Giao diện
        self.label = tk.Label(root, text="Tải dữ liệu (CSV):")
        self.label.pack()
        
        self.upload_button = tk.Button(root, text="Cập nhật", command=self.load_data)
        self.upload_button.pack()
        
        self.train_button = tk.Button(root, text="Huấn luyện", command=self.train)
        self.train_button.pack()
        
        self.predict_button = tk.Button(root, text="Dự đoán", command=self.predict)
        self.predict_button.pack()

        self.database_button = tk.Button(root, text="Database", command=self.open_database_window)
        self.database_button.pack()

        self.hybrid_var = tk.BooleanVar()
        self.hybrid_checkbox = tk.Checkbutton(root, text="Hybrid Models (>= 100 kỳ)", variable=self.hybrid_var)
        self.hybrid_checkbox.pack()

        self.db_window = None  # Biến để lưu cửa sổ "Cập nhật cơ sở dữ liệu"
        
        # Nút "Duyệt thư mục"
        self.browse_button = tk.Button(root, text="Duyệt thư mục", command=self.browse_directory)
        self.browse_button.pack()
        
    def browse_directory(self):
        """Chức năng để chọn thư mục lưu trữ dữ liệu"""
        folder_selected = filedialog.askdirectory()  # Mở hộp thoại chọn thư mục
        if folder_selected:
            self.save_folder = folder_selected
            self.show_info(f"Thư mục đã chọn: {self.save_folder}")
        else:
            self.show_error("Không có thư mục nào được chọn!")

    def load_data(self):
        """Chức năng tải dữ liệu từ file CSV"""
        if self.save_folder is None:
            self.show_error("Vui lòng chọn thư mục để lưu dữ liệu!")
            return
        
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.show_info(f"Dữ liệu đã được tải thành công từ {file_path}")
            except Exception as e:
                self.show_error(f"Không thể tải dữ liệu: {e}")

    def prepare_data(self, data):
        """Chuẩn bị dữ liệu cho mô hình LSTM hoặc Hybrid model"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        X = []
        y = []
        sequence_length = 20  # Dùng 20 kỳ quay trước làm dữ liệu đầu vào
        
        for i in range(len(data) - sequence_length):
            X.append(data_scaled[i:i+sequence_length])
            y.append(data_scaled[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, scaler

    def train_lstm_model(self, X_train, y_train):
        """Huấn luyện mô hình LSTM"""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        return model

    def hybrid_train(self, X_train, y_train):
        """Huấn luyện mô hình Hybrid (LSTM kết hợp với XGBoost)"""
        lstm_model = self.train_lstm_model(X_train, y_train)
        lstm_predictions = lstm_model.predict(X_train)
        
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
        xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
        return lstm_model, xgb_model, lstm_predictions

    def hybrid_predict(self, lstm_model, xgb_model, last_sequence):
        """Dự đoán với mô hình Hybrid"""
        lstm_pred = lstm_model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))
        xgb_pred = xgb_model.predict(last_sequence.reshape(1, -1))
        return lstm_pred[0], xgb_pred[0]

    def train(self):
        """Chức năng huấn luyện mô hình dựa trên dữ liệu"""
        if self.data is None:
            self.show_error("Vui lòng tải dữ liệu trước!")
            return
        
        try:
            X_train, X_test, y_train, y_test, scaler = self.prepare_data(self.data)
            
            if len(self.data) >= 100 and self.hybrid_var.get():
                lstm_model, xgb_model, lstm_predictions = self.hybrid_train(X_train, y_train)
                self.show_info("Hybrid Models đã được huấn luyện!")
            else:
                lstm_model = self.train_lstm_model(X_train, y_train)
                self.show_info("Mô hình LSTM đã được huấn luyện!")
        except Exception as e:
            self.show_error(f"Không thể huấn luyện: {e}")

    def predict(self):
        """Chức năng dự đoán kết quả"""
        if self.data is None:
            self.show_error("Vui lòng tải dữ liệu trước!")
            return
        
        try:
            X_train, X_test, y_train, y_test, scaler = self.prepare_data(self.data)
            last_sequence = X_train[-1]
            
            if len(self.data) >= 100 and self.hybrid_var.get():
                lstm_model, xgb_model, lstm_predictions = self.hybrid_train(X_train, y_train)
                lstm_pred, xgb_pred = self.hybrid_predict(lstm_model, xgb_model, last_sequence)
                predictions = (lstm_pred + xgb_pred) / 2
            else:
                lstm_model = self.train_lstm_model(X_train, y_train)
                predictions = lstm_model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))
            
            predictions = scaler.inverse_transform(predictions)
            
            self.show_info(f"Kết quả dự đoán: {predictions}")
        except Exception as e:
            self.show_error(f"Không thể dự đoán: {e}")

    def open_database_window(self):
        """Mở cửa sổ 'Cập nhật cơ sở dữ liệu'"""
        try:
            if not hasattr(self, 'db_window') or self.db_window is None or not self.db_window.winfo_exists():
                self.db_window = Toplevel(self.root)
                self.db_window.title("Cập nhật cơ sở dữ liệu")

                db = LotteryDatabase(self.db_window)
                db.open_window()
            else:
                self.show_info("Cửa sổ 'Cập nhật cơ sở dữ liệu' đã mở!")
        except Exception as e:
            self.show_error(f"Đã xảy ra lỗi: {e}")

    def show_info(self, message):
        """Hiển thị thông báo thông tin với khả năng sao chép"""
        self.show_message("Thông báo", message)

    def show_error(self, message):
        """Hiển thị thông báo lỗi với khả năng sao chép"""
        self.show_message("Lỗi", message)

    def show_message(self, title, message):
        """Hiển thị cửa sổ thông báo tùy chỉnh có thể sao chép"""
        message_window = Toplevel(self.root)
        message_window.title(title)
        
        text_box = tk.Text(message_window, wrap='word', height=10, width=50)
        text_box.insert(tk.END, message)
        text_box.config(state=tk.DISABLED)  # Không cho phép chỉnh sửa
        text_box.pack(padx=10, pady=10)

        # Thêm nút đóng cửa sổ
        close_button = tk.Button(message_window, text="Đóng", command=message_window.destroy)
        close_button.pack(pady=5)

# Khởi tạo ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = LotteryPredictionApp(root)
    root.mainloop()
