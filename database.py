import pandas as pd
import re

def parse_raw_data_to_training_format(raw_data):
    """
    Chuyển đổi dữ liệu thô thành định dạng CSV chuẩn để huấn luyện mô hình.
    Định dạng đầu ra:
    'Kỳ,Đặc Biệt,Nhất,Nhì,Ba,Bốn,Năm,Sáu,Bảy'
    """
    # Khởi tạo danh sách chứa dữ liệu đã chuẩn hóa
    processed_data = []

    # Tách dữ liệu thành các dòng
    lines = raw_data.split('\n')
    current_record = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Kiểm tra nếu dòng chứa thông tin kỳ quay
        if line.startswith("Kỳ"):
            if current_record:
                # Lưu kỳ quay trước đó vào danh sách
                processed_data.append(current_record)
                current_record = {}

            # Lưu thông tin kỳ quay
            try:
                current_record["Kỳ"] = re.search(r"Kỳ\s+(\S+)", line).group(1)
            except AttributeError:
                print(f"Không thể tách mã kỳ từ dòng: {line}")
                continue
            continue

        # Xử lý thông tin các giải
        match = re.match(r'^(Đặc Biệt|Nhất|Nhì|Ba|Bốn|Năm|Sáu|Bảy)\s+(.+)$', line)
        if match:
            key, value = match.groups()
            # Chuẩn hóa các giá trị
            value = value.replace('\t', '').replace(' ', '').replace('- ', '-')
            current_record[key] = value

    # Thêm bản ghi cuối cùng nếu có
    if current_record:
        processed_data.append(current_record)

    # Tạo DataFrame với các cột chuẩn
    df = pd.DataFrame(processed_data, columns=["Kỳ", "Đặc Biệt", "Nhất", "Nhì", "Ba", "Bốn", "Năm", "Sáu", "Bảy"])
    df.fillna('', inplace=True)  # Điền giá trị trống cho các ô không có dữ liệu
    return df

def save_to_training_csv(data_frame, file_path):
    """
    Lưu dữ liệu đã chuẩn hóa vào file CSV.
    """
    try:
        data_frame.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"Dữ liệu đã được lưu vào {file_path}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")

def process_raw_file_to_csv(input_file_path, output_file_path):
    """
    Đọc file thô, xử lý dữ liệu và lưu vào CSV chuẩn.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            raw_data = file.read()

        # Xử lý dữ liệu
        processed_data = parse_raw_data_to_training_format(raw_data)

        # Lưu dữ liệu ra file CSV
        save_to_training_csv(processed_data, output_file_path)
    except Exception as e:
        print(f"Lỗi khi xử lý file: {e}")
