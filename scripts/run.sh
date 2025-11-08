#!/bin/bash
set -e  # 脚本出错时自动停止，避免继续执行无效步骤

# 1. 激活你的conda虚拟环境（你的环境名是transf，不是默认的transformer-env）
source ~/miniconda3/bin/activate transf  # WSL中conda激活的标准路径

# 2. 安装依赖包（若他人下载你的代码，这步会自动补全依赖）
pip install -r ../requirements.txt  # ../指向transf根目录的requirements.txt

# 3. 运行训练代码（路径对应src/trans_f.py，WSL下相对路径生效）
python ../src/trans_f.py

# 4. 训练完成提示
echo " Transformer训练已完成！"
echo " 模型权重已保存到：../results/checkpoints"
echo " 训练曲线&指标已保存到：../results（training_curve.png + metrics.csv）"