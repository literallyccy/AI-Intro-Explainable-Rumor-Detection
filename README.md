# 可解释的谣言检测最终工程版

本项目用于《人工智能导论》大作业：可解释的谣言检测。

系统实现：
- 输入一条文本；
- 输出二分类结果：`0 = 非谣言`，`1 = 谣言`；
- 输出模型置信度；
- 输出支持“谣言”和支持“非谣言”的关键词/片段；
- 输出模型为什么最终更偏向某一类；
- 可选调用学校大模型接口，将本地证据改写为更自然的判断依据；
- 在 `val.csv` 上计算准确率；
- 保存验证集预测结果和逐条解释结果。

## 1. 方案设计

本项目采用：

```text
本地分类模型 + 特征贡献解释 + 可选大模型解释生成
```

其中：

```text
本地分类模型：TF-IDF + Logistic Regression
本地解释模块：TF-IDF 值 × 逻辑回归权重
大模型接口：只负责把本地证据改写成自然语言，不参与最终分类
```

逻辑回归二分类的决策函数为：

```text
score = intercept + Σ(TF-IDF_i × weight_i)
```

判断规则：

```text
贡献值 > 0：该词或片段支持 1 = 谣言
贡献值 < 0：该词或片段支持 0 = 非谣言
score > 0：模型最终判断为谣言
score < 0：模型最终判断为非谣言
```

## 2. 项目结构

```text
rumor_detection_final_with_llm_report/
├── README.md
├── requirements.txt
├── .env.example
├── report.pdf
├── data/
│   ├── train.csv
│   └── val.csv
├── src/
│   ├── text_utils.py
│   ├── explain.py
│   ├── llm_explain.py
│   ├── train.py
│   ├── predict.py
│   ├── batch_predict.py
│   └── evaluate.py
├── models/
└── results/
```

## 3. 安装依赖

Mac / Linux：

```bash
python3 -m pip install -r requirements.txt
```

Windows：

```bash
python -m pip install -r requirements.txt
```

## 4. 训练模型

```bash
python3 src/train.py
```

训练后生成：
- `models/tfidf_vectorizer.pkl`
- `models/logistic_regression.pkl`
- `results/metrics.json`
- `results/val_predictions.csv`
- `results/val_explanations.csv`
- `results/global_feature_weights.csv`

## 5. 查看评估结果

```bash
python3 src/evaluate.py
```

或：

```bash
cat results/metrics.json
```

## 6. 单条预测

不调用大模型接口：

```bash
python3 src/predict.py "这里输入一条待检测文本"
```

调用大模型接口润色解释：

```bash
python3 src/predict.py "这里输入一条待检测文本" --use-llm
```

如果接口未配置或调用失败，程序会自动退回本地解释。

## 7. 大模型接口配置

本项目不在代码中硬编码 API Key。使用 `.env.example` 作为模板，配置环境变量：

```bash
export SJTU_API_KEY="你的API Key"
export SJTU_API_BASE="接口Base URL"
export SJTU_MODEL_NAME="模型名称"
```

也可以复制：

```bash
cp .env.example .env
```

然后填写 `.env`。注意不要把真实 `.env` 上传到 GitHub。

大模型只根据本地模型给出的标签、置信度、支持谣言/非谣言的特征和决策分数生成自然语言解释，不改变模型分类结果。

## 8. 批量预测

```bash
python3 src/batch_predict.py --input data/val.csv --output results/batch_predictions.csv --text-col text
```

可选使用大模型解释：

```bash
python3 src/batch_predict.py --input data/val.csv --output results/batch_predictions_llm.csv --text-col text --use-llm
```
