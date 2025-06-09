import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt


def process_sentinel2_to_rgb(tif_path, output_path):
    """
    读取哨兵2号多波段TIF文件，提取RGB波段，进行增强和归一化，
    并将其保存为标准的JPG图像。

    Args:
        tif_path (str): 输入的TIF文件路径。
        output_path (str): 输出的JPG文件路径。
    """
    print(f"正在处理文件: {tif_path}")

    # 1. 使用 rasterio 读取 TIF 文件
    try:
        with rasterio.open(tif_path) as src:
            # 根据您的代码，假设波段顺序为：
            # band 1: Blue
            # band 2: Green
            # band 3: Red
            # band 4: NIR
            # band 5: SWIR
            # 我们只需要读取前三个波段 (RGB)
            # rasterio 的波段索引从 1 开始
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)

            print("波段读取成功。")
            print(f"图像尺寸 (高x宽): {src.height} x {src.width}")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 2. 将三个波段堆叠成一个RGB图像
    # np.dstack按照(height, width, channel)的顺序堆叠
    rgb_origin = np.dstack((red, green, blue))

    # --- 优化点：使用百分比截断进行归一化 ---
    # 直接使用0-10000归一化可能效果不佳，因为图像中可能只有很少的像素接近10000。
    # 我们计算2%和98%位置的像素值，忽略最暗和最亮的2%的像素，让色彩分布更均匀。
    p2, p98 = np.percentile(rgb_origin, (2, 98))
    print(f"2%百分位: {p2}, 98%百分位: {p98}")

    # 3. 数据压缩到 0-255
    # 首先将像素值裁剪到 p2 和 p98 之间
    rgb_clipped = np.clip(rgb_origin, p2, p98)

    # 然后将这个范围线性拉伸到 0-255
    rgb_normalized = ((rgb_clipped - p2) / (p98 - p2)) * 255

    # 转换为8位无符号整数，这是图像文件的标准格式
    rgb_final = rgb_normalized.astype(np.uint8)
    print("图像归一化处理完成。")

    # 4. 保存图像
    # OpenCV 保存图像时需要将通道顺序从 RGB 转为 BGR
    bgr_final = cv2.cvtColor(rgb_final, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_final)
    print(f"真彩色图像已成功保存到: {output_path}")

    # 5. 显示图像 (可选)
    # Matplotlib 显示图像需要 RGB 顺序
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_final)
    plt.title('生成的真彩色图像 (True Color Image)')
    plt.axis('off')
    plt.show()


# --- 主程序入口 ---
if __name__ == '__main__':
    # 推荐将文件路径中的反斜杠 \ 全部替换为正斜杠 /
    # 并请确保文件确实存在于这个路径下。

    # 假设您的项目和数据都在 E 盘
    input_tif_path = "E:/python/pythonProject/2019_1101_nofire_B2348_B12_10m_roi.tif"

    # 将输出路径修正到真实存在的 E 盘
    output_jpg_path = "E:/python/pythonProject/output_true_color.jpg"

    # 调用处理函数
    process_sentinel2_to_rgb(input_tif_path, output_jpg_path)