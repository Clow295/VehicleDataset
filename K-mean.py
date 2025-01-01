import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Đọc dữ liệu từ file CSV
df = pd.read_csv(r"D:\Bai Tap\Kho du lieu\Bai_tap_lon\Tieu luan\VehicleDataset\output\car_encoded.csv")

# Hiển thị 5 dòng đầu của dữ liệu
print(df.head())

# Lựa chọn cột liên quan
X = df[['year', 'km_driven', 'selling_price','fuel']]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dùng phương pháp Elbow để tìm số cụm tối ưu
wcss = []  # Within-Cluster Sum of Square (Tổng phương sai trong cụm)
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # inertia_ là tổng khoảng cách bình phương từ điểm tới tâm cụm



# Vẽ biểu đồ Elbow
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, 'bo-')
plt.xlabel('Số cụm (k)')
plt.ylabel('WCSS')
plt.title('Phương pháp Elbow để chọn số cụm tối ưu')
plt.grid(True)
plt.show()

# Thuật toán K-Means

# Số cụm tối ưu
optimal_k = 5

# Huấn luyện K-means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=50)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Xem kết quả cụm
print(df[['selling_price', 'km_driven', 'cluster']].head())

plt.figure(figsize=(10, 6))
plt.scatter(df['km_driven'], df['selling_price'], c=df['cluster'], cmap='viridis')
plt.colorbar(label='Cụm')
plt.xlabel('Km Driven (Km)')
plt.ylabel('Selling Price (M)')
plt.title('K-means Clustering on Vehicle Dataset')

# Chia trục y thành các khoảng nhỏ hơn (ví dụ 1 triệu/lần)
max_price = df['selling_price'].max()
y_ticks = np.arange(0, max_price + 1_000_000, 1_000_000)  # Mỗi lần tăng 1 triệu
plt.yticks(y_ticks, [f"{int(y/1_000_000)}M" for y in y_ticks])  # Hiển thị dưới dạng triệu (M)

plt.grid(True)
plt.show()

score = silhouette_score(X_scaled, df['cluster'])
print(f'Silhouette Score: {score:.2f}')

# Tính số lượng xe trong mỗi cụm
cluster_counts = df['cluster'].value_counts().sort_index()

# Tính tỷ lệ phần trăm
cluster_percentages = (cluster_counts / len(df) * 100).round(2)

# Kết hợp dữ liệu vào DataFrame để hiển thị
cluster_summary = pd.DataFrame({
    'Cluster': cluster_counts.index,
    'Count': cluster_counts.values,
    'Percentage (%)': cluster_percentages.values
})

# In kết quả
print("\n| Cụm | Số lượng xe | Tỷ lệ (%) |")
print("|-----|-------------|-----------|")
for _, row in cluster_summary.iterrows():
    print(f"| {int(row['Cluster'])}   | {row['Count']}    	| {row['Percentage (%)']} %\t|")

# Phân tích kết quả theo cụm
cluster_analysis = df.groupby('cluster').agg(
    year_mean=('year', 'mean'),
    km_driven_mean=('km_driven', 'mean'),
    selling_price_mean=('selling_price', 'mean')
).reset_index()

cluster_analysis['km_driven_mean'] = cluster_analysis['km_driven_mean'].apply(lambda x: f"{x:,.0f} km")
cluster_analysis['selling_price_mean'] = cluster_analysis['selling_price_mean'].apply(lambda x: f"{x:,.0f} $")

print("\n| Cụm | Năm sản xuất (mean) | Số km đã đi (mean)  | Giá bán trung bình  |")
print("|-----|---------------------|---------------------|---------------------|")
for _, row in cluster_analysis.iterrows():
    print(f"| {int(row['cluster']):<3} | {int(row['year_mean']):<19} | {row['km_driven_mean']:<19} | {row['selling_price_mean']:<19} |")

