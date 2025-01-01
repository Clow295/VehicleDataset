import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn file CSV clean
path1 = r"D:\Bai Tap\Kho du lieu\Bai_tap_lon\Tieu luan\VehicleDataset\output\car_clean.csv"

# Load dữ liệu
df1 = pd.read_csv(path1, delimiter=',', nrows=5000)
df1.dataframeName = 'CarDataset.csv'
print(f"Dataset '{df1.dataframeName}' loaded with {df1.shape[0]} rows and {df1.shape[1]} columns.")

# --- HÀM PHÂN PHỐI ---
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    df = df.dropna(axis=1)  # Bỏ cột có giá trị NaN
    nunique = df.nunique()
    df = df[[col for col in df if 1 < nunique[col] < 50]]  # Giới hạn unique values
    if df.empty:
        print("Không có cột đủ điều kiện để vẽ phân phối.")
        return

    nCol = df.shape[1]
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow  # Số dòng biểu đồ
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=60, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        col = df.iloc[:, i]
        if col.dtypes == 'object':  # Nếu là chuỗi
            col.value_counts().plot.bar()
        else:  # Nếu là số
            col.hist()
        plt.title(f'{df.columns[i]}')
        plt.ylabel('Số lượng')
        plt.xticks(rotation=90)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

# --- CHẠY HÀM ---
plotPerColumnDistribution(df1, nGraphShown=10, nGraphPerRow=3)




