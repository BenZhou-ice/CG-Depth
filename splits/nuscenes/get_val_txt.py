import os
import random

# ================== 路径配置 ==================
TRAIN_FILE = '/home/8T/zb/prj/R-MSFM/splits/nuscenes/train_files.txt'
VAL_FILE = '/home/8T/zb/prj/R-MSFM/splits/nuscenes/val_files.txt'


def create_val_from_train():
    # 1. 读取训练集所有行
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ 错误: 找不到训练文件 {TRAIN_FILE}")
        return

    with open(TRAIN_FILE, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    total_train = len(lines)
    if total_train == 0:
        print("⚠️ 训练集为空，无法采样！")
        return

    # 2. 计算 10% 的数量（至少 1 条）
    val_count = max(1, total_train // 10)

    # 3. 随机采样（不重复）
    random.seed(42)  # 固定随机种子，保证每次生成的 val 文件不变
    val_lines = random.sample(lines, val_count)

    # 4. 写入验证集文件
    with open(VAL_FILE, 'w') as f:
        f.write('\n'.join(val_lines))

    print(f"✅ 成功生成验证集文件: {VAL_FILE}")
    print(f"📊 总训练集数据: {total_train} 条")
    print(f"📊 抽取的验证集数据: {val_count} 条 (约 10%)")


if __name__ == '__main__':
    create_val_from_train()