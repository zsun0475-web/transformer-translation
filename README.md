# Transformer 手工实现
本仓库基于 **TensorFlow 2.19.1** 手工实现Transformer模型，用于英中翻译任务，核心代码整合在`src/trans_f.py`中，支持通过`scripts/run.sh`一键启动训练，所有依赖和结果可完全复现。

## 项目结构
```
transf/ # 项目根目录
├── src/ # 核心代码目录
│   └── trans_f.py # 完整代码：含数据加载、Transformer 模型（Encoder/Decoder）、训练逻辑、结果保存
├── data/ # 数据集目录（英中翻译 CSV 文件）
│   ├── train.csv # 训练集（列名：en = 英文原文，zh = 中文译文）
│   └── validation.csv # 验证集（格式同训练集，用于评估模型）
├── results/ # 训练结果输出目录（自动生成）
│   ├── checkpoints/ # 模型权重保存：每 5 个 epoch 自动保存 1 次
│   ├── training_curve.png# 训练 & 验证损失曲线：训练结束后自动生成
│   └── metrics.csv # 训练指标：记录每轮训练损失、验证损失、困惑度
├── scripts/ # 一键运行脚本目录
│   └── run.sh # 一键训练脚本：自动激活环境、安装依赖、启动训练
├── requirements.txt # 依赖包列表：含精确版本，支持一键安装
└── README.md # 项目说明文档：当前你正在看的文件
```

## 环境要求
- 操作系统：Windows（需启用WSL Ubuntu 24.04）/ Linux / macOS
- Python 版本：3.10（你的conda环境版本，避免兼容问题）
- 依赖管理：conda（推荐，创建独立环境）或 pip
- 可选GPU：NVIDIA GPU（需装CUDA Toolkit加速；无GPU则自动用CPU训练）

## 快速复现训练
### 1. 进入项目根目录
打开WSL终端（或VS Code集成终端），执行命令导航到你的项目文件夹（替换成你的实际路径）：
```bash 
cd /mnt/d/大模型/transf
```
### 2. 配置依赖环境
#### 用conda创建独立环境
```bash
conda create -n transf python=3.10 -y
conda activate transf
pip install -r requirements.txt
```
### 3. 启动训练
```bash
cd scripts
chmod +x run.sh
./run.sh
```




### 注意
1. **路径错误**：如果提示“找不到train.csv”，检查`src/trans_f.py`中的数据集路径——WSL用`/mnt/d/大模型/transf/data`，Windows原生环境要改成`D:\大模型\transf\data`。
2. **权限错误**：如果执行`./run.sh`提示“Permission denied”，重新执行`chmod +x run.sh`赋予权限。
3. **GPU问题**：如果终端显示“可用GPU设备: []”，会自动用CPU训练。