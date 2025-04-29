import numpy as np                  # 数值计算库，用于矩阵运算、数组操作等
from sklearn.datasets import fetch_openml  # 数据集获取工具，用于加载MNIST数据集
import matplotlib.pyplot as plt     # 可视化库，用于绘制图像和结果


# -------------------
# 1. 数据加载与预处理
# -------------------
def load_data():
    """
    加载MNIST数据集并进行预处理
    返回：归一化后的图像数据（形状为 [样本数, 28, 28, 1]，像素值范围 [0, 1]）
    """
    mnist = fetch_openml('mnist_784', version=1)  # 从OpenML获取MNIST数据集（784=28x28）
    data = mnist.data.astype(np.float32).values     # 转换为float32类型的NumPy数组
    data = data.reshape(-1, 28, 28, 1)              # 重塑为CNN输入格式：[N, H, W, C]，C=1（单通道）
    data = data / 255.0                             # 像素值归一化到 [0, 1] 范围，便于优化
    return data


# -------------------
# 2. 噪声添加函数
# -------------------
def add_noise(images, noise_factor=0.5):
    """
    向图像添加高斯噪声
    参数：
        images: 输入图像（形状 [N, H, W, C]）
        noise_factor: 噪声强度，控制高斯分布的标准差（均值=0）
    返回：添加噪声并裁剪后的图像（像素值保持在 [0, 1]）
    """
    # 生成与图像形状相同的高斯噪声（均值0，标准差=noise_factor*1.0）
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0., 1.)  # 裁剪像素值到 [0, 1]，避免越界


# -------------------
# 3. 评估指标函数
# -------------------
def psnr(original, denoised):
    """
    计算峰值信噪比（PSNR），衡量像素级失真
    参数：
        original: 原始图像（形状 [H, W, C]）
        denoised: 去噪后的图像（形状同上）
    返回：PSNR值（越高越好，理想值为无穷大）
    """
    mse = np.mean((original - denoised) ** 2)  # 均方误差（MSE）
    if mse == 0:
        return float('inf')  # 无失真时PSNR为无穷大
    max_pixel = 1.0  # 归一化后像素最大值
    return 20 * np.log10(max_pixel / np.sqrt(mse))  # 转换为PSNR（dB）


def ssim(original, denoised):
    """
    简化版结构相似性指数（SSIM），衡量图像结构相似性
    注：实际应用中建议使用更复杂的多尺度实现或成熟库（如skimage）
    参数：同上
    返回：SSIM值（范围 [0, 1]，1表示完全相同）
    """
    max_pixel = 1.0  # 归一化后像素最大值
    mu_x = np.mean(original)    # 原始图像均值
    mu_y = np.mean(denoised)    # 去噪图像均值
    sigma_x = np.std(original)  # 原始图像标准差
    sigma_y = np.std(denoised)  # 去噪图像标准差
    sigma_xy = np.mean((original - mu_x) * (denoised - mu_y))  # 协方差

    c1 = (0.01 * max_pixel) ** 2  # 亮度常数（防止分母为0）
    c2 = (0.03 * max_pixel) ** 2  # 对比度常数

    # 分子：结构相似性的分子部分（亮度、对比度、结构综合项）
    ssim_num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    # 分母：结构相似性的分母部分（原始与去噪图像的统计量乘积）
    ssim_den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)

    return ssim_num / ssim_den  # 返回SSIM值


# -------------------
# 4. 卷积层实现（含前向/反向传播）
# -------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        卷积层初始化
        参数：
            in_channels: 输入通道数（如灰度图为1，RGB为3）
            out_channels: 输出通道数（即卷积核数量）
            kernel_size: 卷积核尺寸（正方形，如3x3）
            stride: 滑动步长
            padding: 填充像素数（保持空间尺寸不变常用same padding，如kernel=3时padding=1）
        """
        # 初始化卷积核（标准差0.1的正态分布，避免梯度爆炸/消失）
        self.kernel = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * 0.1
        self.bias = np.zeros(out_channels)  # 初始化偏置（每个输出通道一个偏置）
        self.stride = stride                # 步长
        self.padding = padding              # 填充
        self.last_input = None              # 存储前向传播的输入，用于反向传播

    def forward(self, x):
        """
        前向传播：计算卷积输出
        参数：x为输入数据（形状 [batch_size, H, W, in_channels]）
        返回：卷积输出（形状 [batch_size, out_H, out_W, out_channels]）
        """
        self.last_input = x  # 保存输入，用于反向传播
        batch_size, h, w, _ = x.shape  # 获取输入尺寸
        k_size = self.kernel.shape[0]  # 卷积核尺寸

        # 填充处理：在图像四周添加0像素，保持空间尺寸或控制边界效应
        x_padded = np.pad(x, ((0, 0), (self.padding, self.padding),
                              (self.padding, self.padding), (0, 0)),
                          mode='constant') if self.padding else x

        # 计算输出尺寸（公式：(H + 2*padding - k_size) // stride + 1）
        out_h = (h + 2 * self.padding - k_size) // self.stride + 1
        out_w = (w + 2 * self.padding - k_size) // self.stride + 1

        output = np.zeros((batch_size, out_h, out_w, self.kernel.shape[-1]))  # 初始化输出

        # 遍历输出特征图的每个位置（i,j）
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride  # 输入区域的行起始位置
                h_end = h_start + k_size   # 输入区域的行结束位置
                w_start = j * self.stride  # 输入区域的列起始位置
                w_end = w_start + k_size   # 输入区域的列结束位置

                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]  # 提取输入切片
                # 计算卷积：输入切片与卷积核的点积，加偏置
                output[:, i, j, :] = np.tensordot(x_slice, self.kernel,
                                                  axes=([1, 2, 3], [0, 1, 2])) + self.bias
        return output

    def backward(self, grad, lr=0.001):
        """
        反向传播：计算梯度并更新参数
        参数：
            grad: 上游梯度（形状 [batch_size, out_H, out_W, out_channels]）
            lr: 学习率，控制参数更新步长
        返回：传递给上游的梯度（输入数据的梯度）
        """
        batch_size = grad.shape[0]  # 批量大小
        k_size = self.kernel.shape[0]  # 卷积核尺寸
        # 对前向传播的输入进行填充（与前向传播逻辑一致）
        x_padded = np.pad(self.last_input,
                          ((0, 0), (self.padding, self.padding),
                           (self.padding, self.padding), (0, 0)),
                          mode='constant') if self.padding else self.last_input

        d_w = np.zeros_like(self.kernel)  # 卷积核梯度
        d_x_padded = np.zeros_like(x_padded)  # 输入数据的梯度（填充后）
        # 旋转卷积核180度（反向传播时卷积核需翻转，等价于相关运算转卷积）
        kernel_rot = np.rot90(self.kernel, 2, axes=(0, 1))

        # 遍历输出特征图的每个位置（i,j），计算梯度
        for i in range(grad.shape[1]):
            for j in range(grad.shape[2]):
                h_start = i * self.stride  # 输入区域的行起始位置（对应前向传播的映射）
                h_end = h_start + k_size
                w_start = j * self.stride
                w_end = w_start + k_size

                if h_end > x_padded.shape[1] or w_end > x_padded.shape[2]:
                    continue  # 超出边界时跳过（理论上由padding保证不会发生）

                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]  # 输入切片
                # 计算卷积核梯度：输入切片与上游梯度的点积，平均到每个样本
                d_w += np.tensordot(x_slice, grad[:, i, j, :], axes=(0, 0)) / batch_size

                # 计算输入梯度：上游梯度与旋转后的卷积核的点积，累加到填充后的输入梯度
                grad_slice = grad[:, i, j, :]  # 提取当前位置的梯度（形状 [batch_size, out_channels]）
                contribution = np.tensordot(grad_slice, kernel_rot,
                                            axes=([1], [3]))  # 形状 [batch_size, k, k, in_channels]
                d_x_padded[:, h_start:h_end, w_start:w_end, :] += contribution

        # 更新参数：卷积核和偏置（偏置梯度为梯度的平均值）
        self.kernel -= lr * d_w
        self.bias -= lr * np.mean(grad, axis=(0, 1, 2))  # 沿批量、高度、宽度维度平均

        # 移除填充：将梯度恢复为输入数据的原始尺寸
        if self.padding:
            return d_x_padded[:, self.padding:-self.padding,
                   self.padding:-self.padding, :]
        return d_x_padded


# -------------------
# 5. ReLU激活函数
# -------------------
class ReLU:
    def __init__(self):
        self.mask = None  # 存储前向传播中大于0的位置，用于反向传播

    def forward(self, x):
        """
        前向传播：ReLU激活（x > 0时输出x，否则0）
        """
        self.mask = (x > 0)  # 记录正值位置（掩码）
        return x * self.mask  # 应用激活

    def backward(self, grad):
        """
        反向传播：梯度仅通过前向传播中为正的位置，负值位置梯度置0
        """
        return grad * self.mask  # 掩码过滤梯度


# -------------------
# 6. 去噪CNN模型（四层卷积结构）
# -------------------
class DenoiseCNN:
    def __init__(self):
        """
        模型结构：
            conv1 -> relu1 -> conv2 -> relu2 -> conv3 -> relu3 -> conv4
        特点：通过增加卷积层深度，提升特征提取能力（浅层提取边缘，深层提取语义）
        """
        # 第一层：输入1通道，输出16通道，3x3卷积，same padding（保持尺寸）
        self.conv1 = Conv2D(1, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU()  # ReLU激活
        # 第二层：输入16通道，输出32通道，3x3卷积，same padding（尺寸不变）
        self.conv2 = Conv2D(16, 32, kernel_size=3, padding=1)
        self.relu2 = ReLU()  # ReLU激活
        # 第三层：输入32通道，输出16通道，3x3卷积，same padding（降通道数，恢复维度）
        self.conv3 = Conv2D(32, 16, kernel_size=3, padding=1)
        self.relu3 = ReLU()  # ReLU激活
        # 第四层：输入16通道，输出1通道（灰度图），3x3卷积，same padding（恢复原始尺寸）
        self.conv4 = Conv2D(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        前向传播流程：按顺序执行各层计算
        """
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.conv4.forward(x)  # 最后一层不加激活，输出像素值
        return x

    def backward(self, grad, lr=0.001):
        """
        反向传播流程：按逆序计算各层梯度（链式法则）
        """
        # 从最后一层开始，逐层反向传播梯度
        grad = self.conv4.backward(grad, lr)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad, lr)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad, lr)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad, lr)
        return grad  # 返回给输入数据的梯度（此处未使用，仅用于链式传播）


# ======================
# 超参数设置（可调节的核心参数）
# ======================
EPOCHS = 30            # 训练轮数：控制模型遍历训练数据的次数，过大可能过拟合，过小可能欠拟合
BATCH_SIZE = 32          # 批量大小：每次训练使用的样本数，影响梯度稳定性和内存占用
LEARNING_RATE = 0.001    # 学习率：控制参数更新的步长，过大易发散，过小收敛慢
NOISE_FACTOR = 0.4       # 噪声因子：控制高斯噪声的强度（标准差=noise_factor*1.0）


# -------------------
# 7. 训练流程
# -------------------
def main():
    # 初始化模型和数据
    model = DenoiseCNN()
    data = load_data()
    train_data = data[:60000]  # 训练集：前60000样本
    test_data = data[60000:]   # 测试集：后10000样本

    # 训练循环
    for epoch in range(EPOCHS):
        epoch_loss = 0
        np.random.shuffle(train_data)  # 打乱训练数据，避免顺序影响优化

        # 按批次处理数据
        for i in range(0, len(train_data), BATCH_SIZE):
            batch_clean = train_data[i:i + BATCH_SIZE]  # 原始干净图像
            batch_noisy = add_noise(batch_clean, NOISE_FACTOR)  # 添加噪声

            # 前向传播：带噪图像输入，得到去噪输出
            output = model.forward(batch_noisy)

            # 计算损失：均方误差（MSE），衡量像素级差异
            loss = np.mean((output - batch_clean) ** 2)
            epoch_loss += loss  # 累加批次损失

            # 反向传播：计算梯度并更新模型参数
            grad = 2 * (output - batch_clean) / BATCH_SIZE  # MSE梯度公式
            model.backward(grad, LEARNING_RATE)  # 反向传播更新权重

        # 打印 epoch 训练结果（平均损失）
        avg_loss = epoch_loss / (len(train_data) // BATCH_SIZE)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # -------------------
    # 8. 模型评估
    # -------------------
    total_psnr = 0
    total_ssim = 0
    # 按批次处理测试数据，避免内存溢出
    for i in range(0, len(test_data), BATCH_SIZE):
        batch_clean = test_data[i:i + BATCH_SIZE]
        batch_noisy = add_noise(batch_clean, NOISE_FACTOR)
        denoised = model.forward(batch_noisy)  # 去噪输出

        # 逐个样本计算指标（因图像形状为 [H, W, C]，需展开维度）
        for j in range(len(batch_clean)):
            total_psnr += psnr(batch_clean[j], denoised[j])
            total_ssim += ssim(batch_clean[j], denoised[j])

    # 计算平均指标
    avg_psnr = total_psnr / len(test_data)
    avg_ssim = total_ssim / len(test_data)
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # -------------------
    # 9. 结果可视化
    # -------------------
    test_samples = data[60000:60005]  # 选取5个测试样本
    noisy_samples = add_noise(test_samples, NOISE_FACTOR)
    denoised = model.forward(noisy_samples)  # 去噪结果

    plt.figure(figsize=(15, 6))  # 创建画布
    titles = ['Original', 'Noisy', 'Denoised']  # 子图标题

    # 绘制3行5列子图（每行对应一种图像：原始/噪声/去噪，每列对应一个样本）
    for col in range(5):  # 样本索引（0-4）
        for row, data_img in enumerate([test_samples, noisy_samples, denoised]):
            plt.subplot(3, 5, row * 5 + col + 1)  # 子图位置
            plt.imshow(data_img[col].squeeze(), cmap='gray')  # 显示单通道图像（挤压维度）
            plt.title(titles[row] if col == 0 else "")  # 仅第一列显示标题
            plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 调整子图间距
    plt.show()  # 显示图像


if __name__ == "__main__":
    main()  # 执行主函数