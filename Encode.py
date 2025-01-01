import pandas as pd
import os

# Đọc dữ liệu từ file
input_file = r"D:\Bai Tap\Kho du lieu\Bai_tap_lon\Tieu luan\VehicleDataset\input\CarDataset.csv"
output_missing_file = r"D:\Bai Tap\Kho du lieu\Bai_tap_lon\Tieu luan\VehicleDataset\output\car_missing.csv"
output_clean_file = r"D:\Bai Tap\Kho du lieu\Bai_tap_lon\Tieu luan\VehicleDataset\output\car_clean.csv"
output_encoded_file = r"D:\Bai Tap\Kho du lieu\Bai_tap_lon\Tieu luan\VehicleDataset\output\car_encoded.csv"

# Bước 1: Đọc dữ liệu
df = pd.read_csv(input_file)

# Bước 2: Danh sách cột cần xử lý
car_cols = ['name', 'year', 'selling_price', 'km_driven','fuel', 'seller_type', 'transmission', 'owner']

# Tạo DataFrame chỉ chứa các cột liên quan
car_df = df[car_cols].copy()

# Bước 3: Tạo DataFrame cho các bản ghi thiếu toàn bộ dữ liệu
car_missing = car_df[car_df.isnull().all(axis=1)]
car_missing.to_csv(output_missing_file, index=False)

# Loại bỏ các bản ghi thiếu dữ liệu
car_clean = car_df.dropna(subset=car_cols, how='all')
car_clean = car_clean[car_clean.isnull().sum(axis=1) <= 7]
car_clean.to_csv(output_clean_file, index=False)

# Bước 4: Ánh xạ mã hóa các giá trị
fuel_mapping = {'CNG': 1, 'Diesel': 2, 'Electric': 3, 'LPG': 4, 'Petrol': 5}
seller_type_mapping = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
transmission_mapping = {'Manual': 1, 'Automatic': 2}
owner_mapping = {
    'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
    'Fourth & Above Owner': 4, 'Test Drive Car': 5
}

# Bước 5: Mã hóa các cột
car_clean['fuel'] = car_clean['fuel'].map(fuel_mapping).fillna(0)
car_clean['seller_type'] = car_clean['seller_type'].map(seller_type_mapping).fillna(0)
car_clean['transmission'] = car_clean['transmission'].map(transmission_mapping).fillna(0)
car_clean['owner'] = car_clean['owner'].map(owner_mapping).fillna(0)

# Bước 6: Lưu file đã mã hóa
car_clean.to_csv(output_encoded_file, index=False)

# In thông tin kết quả
print("Xử lý và mã hóa dữ liệu hoàn tất!")
print(f"Số bản ghi thiếu dữ liệu: {len(car_missing)}")
print(f"Số bản ghi hợp lệ: {len(car_clean)}")
print(f"File đã mã hóa được lưu tại: {output_encoded_file}")
