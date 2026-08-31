import os
import random

# ================== 路径配置 ==================
# 原始 STEPS 生成的夜间场景列表
SOURCE_SPLIT_FILE = '/home/8T/zb/datasets/nu_data/splits/night_train_split.txt'

# 输出到 R-MSFM 框架的 splits 目录
OUTPUT_DIR = '/home/8T/zb/prj/R-MSFM/splits/nuscenes'
TRAIN_OUT_FILE = os.path.join(OUTPUT_DIR, 'train_files.txt')
VAL_OUT_FILE = os.path.join(OUTPUT_DIR, 'val_files.txt')

def generate_fix_split():
    # 1. 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 读取原始场景列表
    if not os.path.exists(SOURCE_SPLIT_FILE):
        print(f"❌ 错误：找不到源文件 {SOURCE_SPLIT_FILE}")
        return

    with open(SOURCE_SPLIT_FILE, 'r') as f:
        # 只取纯场景名，去除空行
        scenes = [line.strip() for line in f.readlines() if line.strip()]

    if not scenes:
        print("⚠️ 警告：源文件中没有任何场景数据！")
        return

    print(f"✅ 从 {SOURCE_SPLIT_FILE} 读取到 {len(scenes)} 个夜间场景。")

    # 3. 随机打乱并切分 9:1
    random.seed(42)  # 固定种子，保证每次结果一致
    random.shuffle(scenes)

    split_idx = int(len(scenes) * 0.9)
    train_scenes = scenes[:split_idx]
    val_scenes = scenes[split_idx:]

    # 4. 写入纯场景名文件（只存 scene-xxxx，不含任何后缀！）
    with open(TRAIN_OUT_FILE, 'w') as f:
        f.write('\n'.join(train_scenes))
    print(f"✅ 已生成训练集: {TRAIN_OUT_FILE} (共 {len(train_scenes)} 个场景)")

    with open(VAL_OUT_FILE, 'w') as f:
        f.write('\n'.join(val_scenes))
    print(f"✅ 已生成验证集: {VAL_OUT_FILE} (共 {len(val_scenes)} 个场景)")

if __name__ == '__main__':
    generate_fix_split()