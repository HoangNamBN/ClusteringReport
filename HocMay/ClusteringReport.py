import pandas as pd
import numpy as np
# thư viện dùng để tính khoảng cách giữa các cặp điểm trong 2 tập hợp một cách hiệu quả
from scipy.spatial.distance import cdist

datas = pd.read_csv("Iris.csv", delimiter=";").drop('Id', axis=1).values
train_full = datas
train = datas[:, :4]
# do tập dữ liệu có nên được chia làm 3 cụm.
n_cluster = 3

# Bước 1: Chọn ngẫn nhiên 3 tâm cho 3 cụm. Mỗi cụm sẽ đại diện bằng một tâm cụm
# Kết quả tra về là của 3 tâm cụmm là
# [[57 38 17 3]
#  [54 39 13 4]
#  [51 37 15 4]]
def Random_cluster(train, n_cluster):
  return train[np.random.choice(train.shape[0], n_cluster, replace=False)]

# Bước 2: Tính khoảng cách giữa các đối tượng đến 3 tâm cụm
#   Loại hoa 1     Loại hoa 2      Loại hoa 3
#       51	          70	          63
#       35            32              33
#       14            47              60
#        2            14              25
# Khoảng cách từ loại Loại hoa 1(51, 35, 14, 2) đến tâm cụm (57,38,17,3) là: 7.41
# Khoảng cách từ loại Loại hoa 1(51, 35, 14, 2) đến tâm (54,39,13,4) là: 5.48
def Choose_cluster(train, centers):
  # Tính toán khoảng cách theo cặp và trung tâm
  D = cdist(train, centers)
  # Chỉ mục trả về của trung tâm gần nhất
  return np.argmin(D, axis= 1)


#Bước 3: Tính lại các toạ độ tâm cho các nhóm mới dựa vào toạ độ các đối tượng trong nhóm bằng đến khi nào k có sự thay đổi nữa thì chuyển sang bước tiếp theo
def Update_centers(train, labels, n_cluster):
  centers = np.zeros((n_cluster, train.shape[1]))
  for k in range(n_cluster):
    # thu thập các điểm được giao cho cụm thứ k
    Xk = train[labels == k, :]
    # Lấy trung bình
    centers[k,:] = np.mean(Xk, axis= 0)
  return centers

# điều kiện dừng của thuật toán: khi không có sự thay đổi của các đối tượng
def Check_update(centers, new_centers):
  return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

# Sử dụng thuật toán Kmeans
def Kmeans(init_centes, init_labels, train, n_cluster):
  centers = init_centes
  labels = init_labels
  times = 0
  while True:
    labels = Choose_cluster(train, centers)
    new_centers = Update_centers(train, labels, n_cluster)
    if Check_update(centers, new_centers):
      break
    centers = new_centers
    times += 1
  return (centers, labels, times)

if __name__ == "__main__":
  init_centers = Random_cluster(train, n_cluster)
  init_labels = np.zeros(train.shape[0])
  centers, labels, times = Kmeans(init_centers, init_labels, train, n_cluster)
  print("Các tâm cụm tìm được là:")
  print(centers)
  print("-------------------------------")
  print('Số lần lặp lại của tâm cụm là:  ', times)
  print("-------------------------------")
  labels = labels.reshape(-1, 1)
  train_full = np.append(train_full, labels, axis=1)
  print("-------------------------------")
  print(train_full)