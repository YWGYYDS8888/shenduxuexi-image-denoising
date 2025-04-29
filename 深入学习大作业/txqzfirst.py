import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data.astype(np.float32).values
    data = data.reshape(-1, 28, 28, 1)
    data = data / 255.0
    return data


def add_noise(images, noise_factor=0.5):
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0., 1.)


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        self.kernel = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * 0.1
        self.bias = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        batch_size, h, w, _ = x.shape
        k_size = self.kernel.shape[0]

        # Padding处理
        x_padded = np.pad(x, ((0, 0), (self.padding, self.padding),
                              (self.padding, self.padding), (0, 0)),
                          mode='constant') if self.padding else x

        # 计算输出尺寸
        out_h = (h + 2 * self.padding - k_size) // self.stride + 1
        out_w = (w + 2 * self.padding - k_size) // self.stride + 1

        output = np.zeros((batch_size, out_h, out_w, self.kernel.shape[-1]))

        # 前向计算
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + k_size
                w_start = j * self.stride
                w_end = w_start + k_size

                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.tensordot(x_slice, self.kernel,
                                                  axes=([1, 2, 3], [0, 1, 2])) + self.bias
        return output

    def backward(self, grad, lr=0.001):
        batch_size = grad.shape[0]
        k_size = self.kernel.shape[0]
        x_padded = np.pad(self.last_input,
                          ((0, 0), (self.padding, self.padding),
                           (self.padding, self.padding), (0, 0)),
                          mode='constant') if self.padding else self.last_input

        # 初始化梯度
        d_w = np.zeros_like(self.kernel)
        d_x_padded = np.zeros_like(x_padded)
        kernel_rot = np.rot90(self.kernel, 2, axes=(0, 1))

        # 反向传播计算
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                # 计算原始输入位置
                h_start = i * self.stride
                h_end = h_start + k_size
                w_start = j * self.stride
                w_end = w_start + k_size

                if h_end > x_padded.shape[1] or w_end > x_padded.shape[2]:
                    continue

                # 权重梯度
                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]
                d_w += np.tensordot(x_slice, grad[:, i, j, :], axes=(0, 0)) / batch_size

                # 输入梯度（关键修正部分）
                grad_slice = grad[:, i, j, :]  # 形状 (batch_size, out_channels)
                contribution = np.tensordot(grad_slice, kernel_rot,
                                            axes=([1], [3]))  # 形状 (batch_size, k, k, in_channels)
                d_x_padded[:, h_start:h_end, w_start:w_end, :] += contribution

        # 参数更新
        self.kernel -= lr * d_w
        self.bias -= lr * np.mean(grad, axis=(0, 1, 2))

        # 移除padding
        if self.padding:
            return d_x_padded[:, self.padding:-self.padding,
                   self.padding:-self.padding, :]
        return d_x_padded


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class DenoiseCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        return x

    def backward(self, grad, lr=0.001):
        grad = self.conv2.backward(grad, lr)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad, lr)
        return grad


# 训练参数
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NOISE_FACTOR = 0.4

# 初始化
model = DenoiseCNN()
data = load_data()
train_data = data[:60000]

# 训练循环
for epoch in range(EPOCHS):
    epoch_loss = 0
    np.random.shuffle(train_data)

    for i in range(0, len(train_data), BATCH_SIZE):
        batch_clean = train_data[i:i + BATCH_SIZE]
        batch_noisy = add_noise(batch_clean, NOISE_FACTOR)

        # 前向传播
        output = model.forward(batch_noisy)

        # 损失计算
        loss = np.mean((output - batch_clean) ** 2)
        epoch_loss += loss

        # 反向传播
        grad = 2 * (output - batch_clean) / BATCH_SIZE
        model.backward(grad, LEARNING_RATE)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / (len(train_data) // BATCH_SIZE):.4f}")

# 结果可视化
test_samples = data[60000:60005]
noisy_samples = add_noise(test_samples, NOISE_FACTOR)
denoised = model.forward(noisy_samples)

plt.figure(figsize=(15, 6))
titles = ['Original', 'Noisy', 'Denoised']
for col in range(5):
    for row, data in enumerate([test_samples, noisy_samples, denoised]):
        plt.subplot(3, 5, row * 5 + col + 1)
        plt.imshow(data[col].squeeze(), cmap='gray')
        plt.title(titles[row] if col == 0 else "")
        plt.axis('off')
plt.tight_layout()
plt.show()