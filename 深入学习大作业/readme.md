本项目实验为大三人工智能专业学生的深度学习大作业，
本项目使用纯NumPy实现了一个包含残差连接的卷积神经网络，用于MNIST手写数字图像去噪任务。通过手动实现卷积层、激活函数和反向传播算法，展示了深度学习核心组件的底层原理。

# MNIST 图像去噪项目

## 一、项目概述
本项目旨在使用卷积神经网络（CNN）对 MNIST 数据集中的手写数字图像进行高斯噪声去除。MNIST 数据集是一个广泛使用的手写数字图像数据集，包含 70,000 张 28×28 像素的灰度图像，其中 60,000 张用于训练，10,000 张用于测试。通过构建一个四层卷积神经网络，我们尝试从受高斯噪声污染的图像中恢复出原始的清晰图像。

## 二、项目背景
在实际应用中，图像在采集、传输过程中容易受到噪声干扰，图像去噪是计算机视觉中的基础任务，能提升图像质量，为后续的图像识别、分析等任务提供更好的输入。选择 MNIST 数据集进行实验，因其格式规范、易于处理，适合作为图像去噪算法的验证平台。

## 三、项目结构
mnist-image-denoising/
├── txqzsecord.py # 主要代码文件
├── README.md # 项目说明文档
plaintext

## 四、环境要求
- Python 3.10
- NumPy
- scikit-learn
- Matplotlib

可以使用以下命令安装所需的 Python 库：
```bash
pip install numpy scikit-learn matplotlib
五、代码说明
1. 数据加载与预处理
python
def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data.astype(np.float32).values
    data = data.reshape(-1, 28, 28, 1)
    data = data / 255.0
    return data
此函数从 OpenML 平台获取 MNIST 数据集，将数据转换为 32 位浮点数类型，重塑为四维张量，并将像素值归一化到 [0, 1] 范围。
2. 噪声添加函数
python
def add_noise(images, noise_factor=0.5):
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0., 1.)
该函数向图像添加高斯噪声，并将像素值裁剪到 [0, 1] 范围，避免越界。
3. 评估指标函数
psnr(original, denoised)：计算峰值信噪比（PSNR），衡量像素级失真。
ssim(original, denoised)：简化版结构相似性指数（SSIM），衡量图像结构相似性。
4. 卷积层和激活函数
Conv2D 类实现了卷积层的前向和反向传播。
ReLU 类实现了 ReLU 激活函数的前向和反向传播。
5. 去噪 CNN 模型
python
class DenoiseCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(16, 32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.conv3 = Conv2D(32, 16, kernel_size=3, padding=1)
        self.relu3 = ReLU()
        self.conv4 = Conv2D(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.conv4.forward(x)
        return x

    def backward(self, grad, lr=0.001):
        grad = self.conv4.backward(grad, lr)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad, lr)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad, lr)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad, lr)
        return grad
该模型包含四层卷积层和三层 ReLU 激活函数，通过前向传播和反向传播进行训练。
6. 训练和评估
在 main 函数中，进行模型的训练和评估，使用均方误差（MSE）作为损失函数，PSNR 和 SSIM 作为评估指标。
六、实验结果
训练损失：随着训练轮数的增加，训练损失逐渐下降，最终达到 0.0106。
测试集指标：平均 PSNR 为 20.0430，平均 SSIM 为 0.9349。
七、总结与展望
本项目通过构建简单的卷积神经网络实现了 MNIST 图像去噪任务，取得了一定的效果。但模型在细节恢复和对强噪声的鲁棒性方面仍有提升空间。未来可以尝试更复杂的网络结构，如 U-Net，引入注意力机制，或进一步调整超参数来优化模型性能。

txqzfirst.py文件为手动构建cnn模型用于图像去噪实验的第一版源代码文件，运行时间约一个小时左右，实验效果去噪效果基本完成，
八、贡献者
YWG（首字母缩写）