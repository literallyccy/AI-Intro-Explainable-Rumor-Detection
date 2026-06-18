# 可解释的谣言检测：本地模型 + 可选大模型辅助判断版

本项目基于原仓库 `literallyccy/AI-Intro-Explainable-Rumor-Detection` 修改而来。原版本中，大模型只用于把本地证据改写成自然语言解释，不参与最终分类。本版本新增 **大模型辅助判断模式**，用户可以选择是否让大模型参与最终标签融合。

## 1. 系统功能

系统输入一条文本，输出：

- 二分类结果：`0 = 非谣言`，`1 = 谣言`；
- 置信度；
- 本地模型支持“谣言”和“非谣言”的词级证据；
- 本地模型解释；
- 可选大模型解释；
- 可选大模型辅助判断结果；
- 本地模型与大模型融合后的最终判断。

## 2. 模型设计

默认情况下，系统使用本地模型进行判断。本地模型采用：

```text
词级 TF-IDF + 字符级 TF-IDF + LinearSVC 硬投票集成
```

其中：

- 词级 TF-IDF 用于生成可读解释；
- 字符级 TF-IDF 用于提高短文本、拼写变化和未知词情况下的泛化能力；
- 多个 LinearSVC 进行硬投票，提高验证集稳定性；
- 保守辟谣规则用于修正明显的“官方辟谣/声明为假/不要传播谣言”文本。

新增的大模型辅助判断模式采用：

```text
本地模型判断 + 本地证据 + 大模型辅助判断 + 融合策略
```

大模型不是默认启用的。只有用户显式加入 `--use-llm-judge` 时，大模型才参与最终判断。

## 3. 项目结构

```text
AI-Intro-Explainable-Rumor-Detection/
├── README.md
├── requirements.txt
├── .env.example
├── LLM_JUDGE_REPORT.md
├── data/
│   ├── train.csv
│   └── val.csv
├── src/
│   ├── text_utils.py
│   ├── rules.py
│   ├── model_utils.py
│   ├── explain.py
│   ├── llm_explain.py
│   ├── train.py
│   ├── predict.py
│   ├── batch_predict.py
│   └── evaluate.py
├── models/
└── results/
```

## 4. 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 5. 训练和评估本地模型

```bash
python src/train.py
python src/evaluate.py
```

训练后生成：

```text
models/rumor_ensemble.pkl
models/model_info.json
results/metrics.json
results/val_predictions.csv
results/val_explanations.csv
results/global_feature_weights.csv
```

## 6. 配置大模型接口

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

填写：

```bash
SJTU_API_KEY="你的 API Key"
SJTU_API_BASE="接口 Base URL，例如 https://xxx/v1"
SJTU_MODEL_NAME="模型名称"
```



## 7. 单条预测

### 7.1 只使用本地模型

```bash
python src/predict.py "Police confirmed that the viral post about contaminated drinking water was false and asked the public not to spread rumors."
```

### 7.2 只让大模型润色解释，不改变分类

```bash
python src/predict.py "A viral message claims that drinking salt water can cure all infections overnight." --use-llm
```

### 7.3 让大模型参与辅助判断

```bash
python src/predict.py "A viral message claims that drinking salt water can cure all infections overnight." --use-llm-judge
```

### 7.4 大模型既参与判断，也生成自然语言解释

```bash
python src/predict.py "A viral message claims that drinking salt water can cure all infections overnight." --use-llm-judge --use-llm
```

## 8. 融合策略

本项目提供三种融合策略：

### conservative：默认策略

```bash
python src/predict.py "文本" --use-llm-judge --llm-fusion conservative
```

本地模型和大模型一致时采用一致结果；二者不一致时，只有在大模型置信度较高且本地模型不是极高置信时，才允许大模型覆盖本地模型。

### override：大模型优先策略

```bash
python src/predict.py "文本" --use-llm-judge --llm-fusion override
```

只要大模型置信度达到阈值，就采用大模型判断。

### agree：一致才增强策略

```bash
python src/predict.py "文本" --use-llm-judge --llm-fusion agree
```

只有本地模型和大模型判断一致时，才提升结果可信度；不一致时保留本地模型结果。

可通过 `--llm-threshold` 设置大模型最低置信度阈值：

```bash
python src/predict.py "文本" --use-llm-judge --llm-threshold 0.8
```

## 9. 批量预测

只使用本地模型：

```bash
python src/batch_predict.py --input data/val.csv --output results/batch_predictions.csv --text-col text
```

批量启用大模型辅助判断：

```bash
python src/batch_predict.py --input data/val.csv --output results/batch_predictions_llm_judge.csv --text-col text --use-llm-judge
```

注意：批量调用大模型会消耗接口额度，并且速度明显慢于本地模型。

## 10. 输出字段说明

启用大模型辅助判断后，输出中会新增：

```text
local_label              本地模型标签
local_confidence         本地模型置信度
llm_judge_enabled        是否启用大模型辅助判断
llm_judge_result         大模型判断结果
llm_fusion_strategy      融合策略
fusion_reason            融合原因
final_source             最终结果来源
```

最终用于展示的标签仍然是：

```text
label
label_name
confidence
```
