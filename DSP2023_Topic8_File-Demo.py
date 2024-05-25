import cv2
import numpy as np
import matplotlib.pyplot as plt


def pca(image, num_components):
    # Bước 1: Chuyển hình ảnh thành ma trận dữ liệu
    image_height, image_width, num_channels = image.shape  # Lấy kích thước hình ảnh
    X = image.reshape(-1, num_channels)  # Biến đổi hình ảnh thành ma trận dữ liệu X

    # Bước 2: Trung tâm hóa dữ liệu
    X_centered = X - np.mean(X, axis=0)

    # Bước 3: Tính ma trận hiệp phương sai
    C = np.cov(X_centered, rowvar=False)

    # Bước 4: Tính trị riêng và vector riêng của ma trận hiệp phương sai
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # Bước 5: Sắp xếp trị riêng và vector riêng theo thứ tự giảm dần
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Bước 6: Tính các thành phần chính
    principal_components = np.dot(X_centered, sorted_eigenvectors[:, :num_components])

    # Bước 7: Chuyển ngược lại thành hình ảnh
    reconstructed_image = np.dot(
        principal_components, sorted_eigenvectors[:, :num_components].T
    )
    reconstructed_image = reconstructed_image + np.mean(X, axis=0)
    reconstructed_image = np.uint8(
        reconstructed_image.reshape(image_height, image_width, num_channels)
    )

    return principal_components, reconstructed_image


# Đọc hình ảnh từ file
image_path = "OIP.jpeg"
image = cv2.imread(image_path)

# Số lượng thành phần chính cần giữ lại
num_components = 2

# Thực hiện PCA và lấy hình ảnh đã tái tạo
principal_components, reconstructed_image = pca(image, num_components)

# In ra các thành phần chính
print("Các thành phần chính:")
print(principal_components)

# Hiển thị hình ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Hình ảnh gốc")
plt.axis("off")

# Hiển thị hình ảnh tái tạo từ PCA
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
plt.title("Hình ảnh tái tạo từ PCA")
plt.axis("off")

plt.show()
