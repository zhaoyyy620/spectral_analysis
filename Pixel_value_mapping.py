#%%
import os
import joblib
import spectral
import cv2
import numpy as np
import torch
from skimage import measure
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.colors import Normalize

from Regression.CnnModel import LPCNet

hdr_file_path = r'' #加载原始数据，后缀为.hdr



# 输出CSV文件的路径
csv_file_path = ''

# 准备写入CSV文件
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)

    # 遍历指定文件夹下的所有.hdr文件
    if hdr_file_path.endswith('.hdr'):
            # 读取高光谱图像
            img = spectral.open_image(hdr_file_path)
            data = img.load()
            # print(data.shape)
#---------------------------------------------------------------------------------------------------------------------------------#%%
            # 计算每个波段的标准差
            std_devs = np.array([np.std(data[:, :, band]) for band in range(data.shape[2])])
            # 找出标准差最大的波段
            max_std_dev_band = np.argmax(std_devs)
            print(f"颜色差异最大的波段是: {max_std_dev_band}, 标准差为: {std_devs[max_std_dev_band]}")
            # Show the picture from hyperspectral images

            img_show = data[:, :, 160]
            # 将像素范围规范到0-255
            img_show = cv2.normalize(img_show, None, 0, 255, cv2.NORM_MINMAX)
            plt.imshow(img_show)
            plt.show()

            # 提取一个波段的图像
            band_image = data[:, :,160]

            # 将图像数据进行归一化，将像素值范围缩放到0-255之间
            normalized_image = cv2.normalize(band_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            # 使用大津法获取阈值
            thresh, _ = cv2.threshold(normalized_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # 获取掩膜图像并显示
            mask_image = cv2.threshold(normalized_image, thresh, 255, cv2.THRESH_BINARY)[1]

            #开运算处理
            # 定义核大小和形状
            kernel_size = (10, 6)
            kernel_shape = cv2.MORPH_CROSS
            kernel = cv2.getStructuringElement(kernel_shape, kernel_size)

            # 进行闭运算
            mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

            # 膨胀操作
            mask_image = cv2.dilate(mask_image, kernel, iterations=1)

            #用于三维分割
            mask_image_2 = mask_image

            # 提取掩膜图像的连通区域，并对联通区域进行编号
            num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
            # 对连通区域进行编号
            for label in range(1, num_labels):
                # 获取连通区域的中心坐标
                center_x, center_y = centroids[label]
                # 将连通区域的编号显示在中心位置
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (125, 125, 125)  # 黑色
                thickness = 1
                cv2.putText(mask_image, str(label), (int(center_x), int(center_y)), font, font_scale, font_color,
                            thickness, cv2.LINE_AA)
            plt.figure(dpi=300)
            plt.imshow(mask_image)
            plt.show()

            # 找出面积最大的连通区域编号
            max_area_label = np.argmax(stats[1:, -1]) + 1  # 加1是因为排除了背景

            # 获取面积最大的连通区域的掩膜
            region_mask = (labeled_img == max_area_label).astype(np.uint8)

            # 使用掩膜提取区域数据
            region_spectra = cv2.bitwise_and(data, data, mask=region_mask)

            # 数据的波段数或光谱维度
            z_dims = data.shape[-1]

            # 将区域数据重塑为二维数组，每行代表一个像素点的光谱数据
            reshaped_region_spectra = np.reshape(region_spectra, (-1, z_dims))

            # 过滤掉全为0的行，这些行代表掩膜之外的像素点
            filtered_region_spectra = reshaped_region_spectra[np.any(reshaped_region_spectra != 0, axis=1)]
            print(filtered_region_spectra.shape)  #256

            # 将每个像素点的光谱反射率写入CSV文件
            for spectra in filtered_region_spectra:
                writer.writerow([os.path.basename(hdr_file_path)] + list(spectra))

            # 可视化光谱曲线（这里可能需要修改逻辑以适应单个像素点的情况）
            plt.figure(figsize=(10, 6))
            for spectra in filtered_region_spectra:  # 这可能需要进一步的逻辑来选择特定的光谱或平均光谱
                plt.plot(spectra, alpha=0.1)  # 设置透明度，以便能看到重叠的部分
            plt.xlabel('Wavelength Band')
            plt.ylabel('Intensity')
            plt.title(f'Spectral Curves of All Pixels in the Largest Region {max_area_label}')
            plt.legend([f'Region {max_area_label}'])

            # 在图中标注最大连通区域的编号
            plt.annotate(f'Max Area Region: {max_area_label}',
                         xy=(0.5, 0.9), xycoords='axes fraction',
                         textcoords='offset points', color='red')
            plt.show()


#------------------------------------黑白校正（可选）---------------------------------------------#
# 定义读取黑白背景数据的函数
def read_background_data(file_path):
    return pd.read_csv(file_path, header=None).values.squeeze()

# 定义光谱数据校正函数
def correct_spectrum(spectra, black_background, white_background):
    return (spectra - black_background) / (white_background - black_background)

# 读取黑白背景校正数据，如果在软件中校正，这一步就不需要
# 如果通过公式校正，这一步就必须
black_background_path = 'black_average_spectrum.csv'
white_background_path = 'white_average_spectrum.csv'
black_background = read_background_data(black_background_path)
white_background = read_background_data(white_background_path)

# 对每个像素点的光谱反射率进行校正
corrected_spectra = correct_spectrum(filtered_region_spectra, black_background, white_background)

# # 展示校正后的所有光谱曲线
# plt.figure(figsize=(10, 8))
# for row in corrected_spectra:
#     plt.plot(row, alpha=0.2)  # 使用alpha值来控制曲线的透明度以便更好地观察重叠
#
# plt.xlabel('Wavelength Index')
# plt.ylabel('Corrected Intensity')
# plt.title('Corrected Spectra')
# plt.show()
#------------------------------------end---------------------------------------------#

# 加载模型
model_path = r''  #加载模型，后缀为.pth
model = LPCNet()
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载标准化器
scaler_path = r'standscale.pkl'
scaler = joblib.load(scaler_path)
yscaler_path = r'yscaler.pkl'
yscaler = joblib.load(yscaler_path)

# 输入数据
input_data = corrected_spectra[:, 1:]

# 标准化输入数据
input_data_scaled = scaler.transform(input_data)

# 将输入数据转换为 torch 张量，并增加一个维度
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).unsqueeze(1)

# 预测光谱反射率的真实值
with torch.no_grad():
    y_pred_scaled = model(input_tensor).numpy()

# 反标准化预测值
y_pred = yscaler.inverse_transform(y_pred_scaled)

# 检查预测值的形状和范围
print(f"Prediction shape: {y_pred.shape}")
print(f"Prediction range: {y_pred.min()} - {y_pred.max()}")

# 创建 CSV 文件并写入预测值
predicted_values_csv_path = os.path.join(os.getcwd(), 'predicted_values_car.csv')

with open(predicted_values_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Pixel Index', 'Predicted Value'])
    for idx, value in enumerate(y_pred):
        x, y = np.column_stack(np.where(region_mask > 0))[idx]
        writer.writerow([f'({x},{y})', value[0]])

print(f"预测值已保存到：{predicted_values_csv_path}")


# 计算预测值的最大值和最小值
min_val = np.min(y_pred)
print(min_val)
max_val = np.max(y_pred)
print(max_val)

# # 向下取整和向上取整到最近的整数
# min_val_rounded = int(np.floor(min_val))
# max_val_rounded = int(np.ceil(max_val))

# 向下取整和向上取整到小数点后两位
min_val_rounded = round(np.floor(min_val * 100) / 100, 2)
max_val_rounded = round(np.ceil(max_val * 100) / 100, 2)

# 创建一个与mask_image相同大小的全黑背景图像
black_background = np.zeros((775, 696), dtype=np.float32)  #与你光谱图像分辨率大小有关

# 使用MinMaxScaler来将预测值缩放到实际的预测值范围
scaler = MinMaxScaler(feature_range=(min_val_rounded, max_val_rounded))
y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1)).reshape(-1)

# 获取掩膜区域的坐标
mask_coords = np.column_stack(np.where(region_mask > 0))

# 初始化黑色背景图像
black_background = np.zeros(region_mask.shape, dtype=np.float32)

# 将缩放后的预测值填充到对应的像素位置
for (value, (x, y)) in zip(y_pred_scaled, mask_coords):
    black_background[x, y] = value

# 创建一个masked array，其中背景（即值为0的区域）不会被绘制
masked_image = np.ma.masked_where(black_background == 0, black_background)

# 设置一个colormap，并将不需要绘制的颜色（masked values）设置为黑色
cmap = plt.cm.jet
cmap.set_bad(color='black')

# 设置正常化的范围以及颜色条的范围，确保范围是整数
norm = Normalize(vmin=min_val_rounded, vmax=max_val_rounded)

# 绘制图像
plt.figure(figsize=(10, 7))
im = plt.imshow(masked_image, cmap=cmap, norm=norm)
cbar = plt.colorbar(im)
cbar.set_label('Estimated values (mg/g)', fontsize=22, fontweight="normal", fontname="Times New Roman")
cbar.ax.tick_params(labelsize=16)  # 设置比色板的字体大小
plt.title('estimated values mapping', fontsize=26, fontweight="normal", fontname="Times New Roman")
plt.axis('off')  # 不显示坐标轴

# 保存图像为500dpi
output_image_path = os.path.join(os.getcwd(), 'pic.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

plt.show()
