import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# ==============================================================================
# 1. 配置参数 (你可以根据需要修改这里)
# ==============================================================================
# --- 输入/输出路径 ---
# 脚本会自动在当前目录下寻找这两个文件夹
INPUT_IMAGE_DIR = "pic1/"
INPUT_MASK_DIR = "masks_2/"

# 脚本会自动创建这两个文件夹来存放增强后的结果
OUTPUT_IMAGE_DIR = "augmented_images/"
OUTPUT_MASK_DIR = "augmented_masks/"

# --- 增强参数 ---
# 为每一张原始图片生成多少张增强后的图片
# 如果想最终得到50张，已有10张，这里就填 (50-10)/10 = 4
NUM_AUGMENTATIONS_PER_IMAGE = 9  # 我们增强到100张，每张原始图生成9张新的

# ==============================================================================
# 2. 定义强大的数据增强“流水线”
# ==============================================================================
# Compose将多个增强操作串联起来。p=0.5表示每个操作有50%的概率被应用。
# 这使得每次生成的图片都是独一无二的。
transform_pipeline = A.Compose([
    # --- 几何变换 ---
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.0625,  # 平移范围
        scale_limit=0.1,  # 缩放范围
        rotate_limit=45,  # 旋转范围
        p=0.8,
        border_mode=cv2.BORDER_CONSTANT,  # 旋转后的填充模式
        value=0,  # 填充值 (黑色)
        mask_value=0  # Mask的填充值 (黑色)
    ),

    # --- 弹性形变 (王牌！) ---
    A.ElasticTransform(
        p=0.5,
        alpha=120,
        sigma=120 * 0.05,
        alpha_affine=120 * 0.03,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0
    ),
    A.GridDistortion(p=0.3),

    # --- 色彩与光照 ---
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),

    # --- 噪声与模糊 ---
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=7, p=0.3),
])


# ==============================================================================
# 3. 主执行函数
# ==============================================================================
def main():
    """
    主函数：读取原始数据，应用增强并保存结果。
    """
    print("--- 开始数据增强流程 ---")

    # 安全地创建输出文件夹
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    # 查找所有原始图片
    image_filenames = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_filenames:
        print(f"错误：在 '{INPUT_IMAGE_DIR}' 文件夹中没有找到任何图片文件。请检查路径。")
        return

    print(f"共找到 {len(image_filenames)} 张原始图片。将为每张图片生成 {NUM_AUGMENTATIONS_PER_IMAGE} 个增强版本。")

    # 使用tqdm创建进度条
    for filename in tqdm(image_filenames, desc="增强进度"):
        # 构建路径
        img_path = os.path.join(INPUT_IMAGE_DIR, filename)

        # 构建mask路径，并智能处理扩展名不匹配的问题
        base_name, _ = os.path.splitext(filename)
        mask_filename = f"{base_name}.png" # 假设mask总是.png格式
        mask_path = os.path.join(INPUT_MASK_DIR, mask_filename)

        # 检查对应的Mask是否存在
        if not os.path.exists(mask_path):
            print(f"\n[警告] 找不到图片 '{filename}' 对应的Mask，已跳过此文件。")
            continue

        # 读取原始图像和Mask
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 确认读取成功
            if image is None:
                print(f"\n[警告] 无法读取图片 '{img_path}'，可能文件已损坏。已跳过。")
                continue
            if mask is None:
                print(f"\n[警告] 无法读取Mask '{mask_path}'，可能文件已损坏。已跳过。")
                continue

        except Exception as e:
            print(f"\n[错误] 读取文件 '{filename}' 时发生异常: {e}。已跳过。")
            continue

        # 循环生成增强样本
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            # 应用定义好的增强流水线
            augmented = transform_pipeline(image=image, mask=mask)

            aug_image = augmented['image']
            aug_mask = augmented['mask']

            # 构建新的文件名
            base_name, extension = os.path.splitext(filename)
            new_img_filename = f"{base_name}_aug_{i + 1}{extension}"
            new_mask_filename = f"{base_name}_aug_{i + 1}.png" # 总是将mask保存为png

            # 保存新的图像和Mask
            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, new_img_filename), aug_image)
            cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, new_mask_filename), aug_mask)

    print("\n--- 数据增强全部完成！---")
    total_generated = len(image_filenames) * NUM_AUGMENTATIONS_PER_IMAGE
    print(f"成功生成了 {total_generated} 组新的图像和Mask。")
    print(f"增强后的图片位于: '{OUTPUT_IMAGE_DIR}'")
    print(f"增强后的Mask位于: '{OUTPUT_MASK_DIR}'")
    print("\n下一步提示：请将原始数据和增强后的数据合并，用于模型训练。")


if __name__ == "__main__":
    main()