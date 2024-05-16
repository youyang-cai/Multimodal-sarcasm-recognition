import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import PIL.Image
import os
import pickle

# 设置使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载预训练的 ResNet-152 模型
weights = models.resnet152(pretrained=True)
# 移除全连接层，保留到第二个到最后的层，输出2048维特征
weights = nn.Sequential(*list(weights.children())[:-1])
model = nn.DataParallel(weights).to(device)
model.eval()

# 定义图像处理变换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义提取视觉向量的函数
def get_visual_vector(filename):
    img = PIL.Image.open(filename)
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)  # 将图像数据移动到GPU上
    with torch.no_grad():  # 在推理时不要进行梯度计算，以节省内存
        vector = model(img_tensor)
    # 移除一个维度，因为输出将是[1, 2048, 1, 1]
    return vector.squeeze()

# 处理目录中的所有图片
features = {}
filelist = os.listdir("./picture")
for filename in filelist:
    full_path = os.path.join("./picture", filename)
    visual_feature = {'visual': get_visual_vector(full_path)}
    visual_name_noext = os.path.splitext(filename)[0]
    features[visual_name_noext] = visual_feature
    torch.cuda.empty_cache()  # 释放不需要的缓存

# 将提取的特征保存到文件
with open('visual_features.pkl', 'wb') as f:
    pickle.dump(features, f)
