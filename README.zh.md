[English](README.md) | [简体中文](README.zh.md)

# Mini NanoGPT 🚀

## 这是初始版本的 *Mini nanoGPT* 项目，现已不再维护。

**新版具有更现代化和实时的损失曲线绘制 UI 和能力，这也让模型的训练流程时间大幅减小。**

**同时，新版引入了数据库来存储信息，让模型更易于管理，并能储存历史训练信息，随时切换加载。**

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone --depth 1 --branch old https://github.com/ystemsrx/mini-nanoGPT.git
cd mini-nanogpt

# 安装依赖（Python 3.7+）
pip install -r requirements.txt
```

### 2. 启动项目
```bash
python main.py
```
打开浏览器访问显示的链接，就能看到训练界面了！（一般是 http://localhost:7860）

## 🎮 使用指南

### 第一步：准备数据
- 打开"数据处理"页面，选择或粘贴你的训练文本并选择分词方式。若要追求更好的效果，可以勾选使用分词器，会自动根据你的文本内容构建词汇表。
- 如果你暂时不想使用验证集，可以勾选“暂不使用验证集”。
- 完成后点击"开始处理"。
这里我用一小段文本来举例：

![image](https://github.com/user-attachments/assets/ec8db0d6-5673-43ae-a4cb-ac064f7209ae)


### 第二步：训练模型
- 切换到"训练"页面，根据需要调整参数（如果只是想体验，可以保持默认值）。
- 程序支持实时显示训练集和验证集的损失曲线。如果在第一步中你生成了验证集，理论上下方损失曲线处会出现两条，蓝色为训练集损失曲线，橙色为验证集损失曲线。
- 如果只显示了1条曲线，请检查终端输出，如果有类似这样的输出
  ```
  Error while evaluating val loss: Dataset too small: minimum dataset(val) size is 147, but block size is 512. Either reduce block size or add more data.
  ```
  说明你设置的block size比你的验证集大，请将它的大小调小，例如128。
- 这样你应当能够正常的看到两条动态变化的曲线。
- 点击"开始训练"，等待模型训练完成

![image](https://github.com/user-attachments/assets/75e53570-393b-48db-aac3-f9b6822d05b1)


#### 仅评估模式？
- 这个模式能够让你评估模型在验证集上的损失。请将`评估种子数量 (Number of Evaluation Seeds)`调为大于0的任意值，将开启仅评估模式。你能看到模型在使用不同种子上的损失。

### 第三步：生成文本
1. 进入"推理"页面
2. 输入一段开头文字
3. 点击"生成"，看看模型会写出什么！

![image](https://github.com/user-attachments/assets/5f985e89-d7c2-4f3a-9500-5713497148cd)

## 📁 项目结构
```
mini-nanogpt/
├── main.py          # 启动程序
├── config/          # 配置文件
├── data/            # 数据处理和存储
├── modules/         # 模型定义、推理生成、接口相关
└── trainer/         # 训练相关
```

## 📝 许可证
本项目采用 [MIT License](LICENSE) 协议开源。

---

🎉 **开始你的 GPT 之旅吧！**
