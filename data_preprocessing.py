import pandas as pd
import numpy as np

def prepare_data(file_path, time_steps=10):
    """
    Chuyển đổi dữ liệu đầu vào thành chuỗi thời gian.
    :param file_path: Đường dẫn đến file CSV chứa dữ liệu xổ số
    :param time_steps: Số bước thời gian cho LSTM
    :return: Dữ liệu đầu vào X và nhãn y
    """
    data = pd.read_csv(file_path)
    numerical_data = data.iloc[:, 1:].applymap(
        lambda x: [int(num) for num in x.split('-') if num.isdigit()]
    )
    sequences = numerical_data.to_numpy()
    
    X, y = [], []
    for i in range(len(sequences) - time_steps):
        X.append(sequences[i:i+time_steps])
        y.append(sequences[i+time_steps])
    
    return np.array(X), np.array(y)
