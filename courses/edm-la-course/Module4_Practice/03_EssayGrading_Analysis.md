## 习题三：作文数据分析与分类

**背景：** 你拥有1500条作文数据。每条数据包含已计算好的特征、作文质量的标签（例如“优秀”、“良好”、“合格”、“待改进”——需要明确标签是什么），以及作文的原始文本。
**已计算特征：** 得分、长度、不重复词的数量、形容词占比、动词占比、名词占比、句子个数、语法错误个数。
**要求：**
1.  建立作文质量预测的分类模型并评估其性能。
2.  分析模型结果，识别影响作文质量评价的关键因素。
3.  撰写分析报告。
4.  （选做）有能力的同学可以尝试从原始文本中提取新的特征。

---

### 第一部分：理清思路 —— 我们要怎么做？

1.  **理解任务目标：**
    * **核心：** 建立一个**分类模型**来预测作文的质量等级（标签）。
    * **关键：**
        * 选择合适的分类算法。
        * 有效地利用已提供的8个结构化特征。
        * （选做）探索从原始文本中提取更多信息。
        * 评估模型性能。
        * 从模型中洞察哪些特征对作文质量影响较大。

2.  **数据理解与准备：**
    * **输入数据：** 1500条记录。
        * **特征 (Features)：** 8个已计算好的数值型特征。
        * **标签 (Target Variable)：** 作文质量等级（这是我们要预测的）。首先要明确标签的具体类别和分布情况 (e.g., "优秀", "良好", "合格" – 是二分类、三分类还是更多？各类别样本是否均衡？)。
        * **原始文本 (Raw Text)：** 用于选做任务，提取新特征。
    * **数据探索 (EDA)：**
        * 查看各特征的统计描述（均值、方差、分布等）。
        * 分析特征与作文质量标签之间的初步关系（例如，高质量作文的平均长度是否更长？语法错误是否更少？）。
        * 可视化特征分布和特征间的关系。

3.  **数据预处理 (针对已计算特征)：**
    * **缺失值处理：** 检查8个特征及标签是否有缺失值，并选择合适的处理策略（如删除、填充）。
    * **特征标准化/归一化：** 对于某些分类算法（如基于距离的SVM、KNN，或使用正则化的逻辑回归），对数值型特征进行标准化或归一化是很重要的，可以避免某些特征因数值范围大而主导模型。
    * **标签编码：** 如果作文质量标签是文本（如“优秀”、“良好”），需要将其转换为数值编码（如0, 1, 2...）才能被大多数分类算法使用。

4.  **（选做）从原始文本中提取新特征：**
    * 这部分可以大大提升模型的性能和分析的深度。
    * **思路（回顾“方法篇-第五讲”文本挖掘部分）：**
        * **文本预处理：** 分词、去除停用词、词形还原/词干提取。
        * **简单文本统计特征：** 平均句长、平均词长、不同词性（如副词、连词）占比、特定关键词/短语频率。
        * **可读性指标：** 如 Flesch Reading Ease, Gunning Fog Index (需要特定库或公式实现)。
        * **（更高级）词袋模型 (Bag-of-Words) / TF-IDF 特征：** 将文本转换为高维的词频或TF-IDF向量。这会产生很多新特征。
        * **（更高级）词嵌入 (Word Embeddings like Word2Vec, GloVe, FastText) 或句嵌入：** 将词或句子映射到低维稠密的向量空间，能捕捉语义信息。
    * 如果提取了新特征，需要将它们与原有的8个特征合并。

5.  **分类模型建立与评估：**
    * **数据划分：** 将数据集划分为**训练集 (Training Set)** 和**测试集 (Test Set)** (例如，80%训练，20%测试；或使用交叉验证)。模型在训练集上学习，在测试集上评估其泛化能力。
    * **选择分类算法（回顾“方法篇-第二讲”）：**
        * 逻辑回归 (Logistic Regression)
        * 决策树 (Decision Trees)
        * 随机森林 (Random Forest) - 通常比单棵决策树效果好且不易过拟合。
        * 支持向量机 (Support Vector Machines, SVM)
        * 朴素贝叶斯 (Naive Bayes)
        * K-最近邻 (K-Nearest Neighbors, KNN)
        * （更高级）梯度提升机 (Gradient Boosting Machines, e.g., XGBoost, LightGBM)
        * **建议：** 可以先尝试几种相对简单的模型（如逻辑回归、决策树），再尝试更强大的模型（如随机森林）。
    * **模型训练：** 用训练集数据拟合所选的分类算法。
    * **模型评估（使用测试集）：**
        * **混淆矩阵 (Confusion Matrix)：** 展示模型对每个类别的预测情况（TP, FP, TN, FN）。
        * **准确率 (Accuracy)：** 整体预测正确的比例。
        * **精确率 (Precision)：** 预测为某类别的样本中，真正是该类别的比例（“查准”）。
        * **召回率 (Recall / Sensitivity)：** 某类别所有真实样本中，被正确预测出来的比例（“查全”）。
        * **F1分数 (F1-Score)：** 精确率和召回率的调和平均值。
        * **AUC-ROC 曲线 (Area Under the ROC Curve)：** （主要用于二分类，也可扩展到多分类）衡量模型在不同阈值下区分正负样本的能力。
        * **注意类别不平衡问题：** 如果不同质量等级的作文数量差异很大，单纯看准确率可能会有误导，此时应更关注精确率、召回率、F1分数或平衡准确率。

6.  **结果分析与应用：**
    * **分析分类模型的结果：** 哪些类别的作文更容易被正确分类？哪些更容易被混淆？（结合混淆矩阵分析）。
    * **识别影响作文质量评价的关键因素：**
        * **特征重要性 (Feature Importance)：** 许多模型（如决策树、随机森林、梯度提升机）可以直接输出各个特征对预测的贡献程度。
        * **模型系数 (Model Coefficients)：** 对于线性模型（如逻辑回归），可以查看系数的大小和正负来理解特征的影响方向和强度（需注意特征已标准化的前提）。
        * 通过分析这些重要特征，可以理解哪些方面（如词汇丰富度、句子结构、语法准确性、篇章长度等）对作文质量的评价影响更大。

7.  **报告撰写：**
    * 结构清晰，包含引言、数据描述与预处理、模型选择与构建过程、性能评估结果、关键特征分析、结论与讨论（包括模型的局限性和未来改进方向）。

---

### 第二部分：小步子任务单 —— 完成作业的推荐步骤

1.  **数据加载与探索性数据分析 (EDA)：**
    * 用 `pandas` 加载包含已计算特征和标签的数据。
    * 查看数据维度、特征类型、缺失值情况。
    * 分析标签的分布（各类作文数量是否均衡）。
    * 计算各特征的描述性统计量。
    * 可视化特征的分布（如直方图、箱形图），以及特征与标签之间的关系（如按标签分组后，各特征的箱形图）。

2.  **数据预处理：**
    * **任务2.1：处理缺失值**（如果存在）。
    * **任务2.2：标签编码** (将文本标签如“优秀”转换为0, 1, 2...)。
    * **任务2.3：特征标准化/归一化** (对8个数值型特征进行处理)。

3.  **（选做）新特征提取（基于原始文本）：**
    * 如果进行此步骤，需要：
        * 加载原始文本数据。
        * 进行文本预处理（分词等）。
        * 提取你认为有用的新特征（如平均句长、TF-IDF等）。
        * 将新特征与原有特征合并。之后的数据划分和模型训练都将基于这个扩展后的特征集。

4.  **模型训练与评估：**
    * **任务4.1：划分训练集和测试集** (例如，80%训练，20%测试，注意设置 `random_state` 以便结果可复现)。
    * **任务4.2：选择至少两种分类算法进行尝试。**
        * 例如，先用逻辑回归或决策树作为基线模型。
        * 再尝试随机森林或梯度提升等更强大的模型。
    * **任务4.3：在训练集上训练模型。**
    * **任务4.4：在测试集上进行预测，并评估模型性能。**
        * 计算准确率、精确率、召回率、F1分数（注意多分类情况下，这些指标可以是宏平均、微平均或加权平均）。
        * 绘制并分析混淆矩阵。
        * （可选）绘制ROC曲线并计算AUC值（如果适用）。
    * **任务4.5：（可选）模型调优。** 可以尝试调整模型的超参数（如决策树的深度、随机森林的树的数量等）来提升性能，或使用交叉验证来更稳健地评估模型。

5.  **结果分析与解释：**
    * **任务5.1：比较不同模型的性能，选择一个或几个表现较好的模型进行深入分析。**
    * **任务5.2：分析模型的预测错误情况**（哪些类别的作文容易被误判？误判成了什么类别？）。
    * **任务5.3：提取并解释特征重要性。** 识别出对作文质量等级预测贡献最大的几个特征，并结合教育学知识进行解读。

6.  **撰写分析报告：**
    * 按照要求，清晰、完整地呈现整个分析过程、结果和发现。
    * 使用图表辅助说明。

---

### 第三部分：参考答案（思路与关键Python代码片段提示）

**重要声明：** 以下代码片段仅为功能演示，并非完整解决方案。你需要根据实际数据和分析需求进行调整。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载与探索 (假设数据在 data.csv，标签列名为 'quality_label')
# df = pd.read_csv('data.csv')
# print(df.head())
# print(df.info())
# print(df['quality_label'].value_counts()) # 查看标签分布

# 假设特征列是 '得分', '长度', ..., '语法错误个数'
# feature_columns = ['长度', '不重复词的数量', '形容词占比', '动词占比', '名词占比', '句子个数', '语法错误个数'] 
# # '得分'通常是人工打分的结果，如果是作为预测目标的一部分，则不应作为输入特征。
# # 如果'得分'是已有的机器评分或人工评分，而'quality_label'是基于'得分'划分的等级，也需要注意。
# # 这里假设 '得分' 是一个可以用来预测 'quality_label' 的特征之一，或者 'quality_label' 不是由 '得分' 直接转换的。
# # 如果 '得分' 本身就是我们要预测的（回归问题），或者 'quality_label' 是 '得分' 的直接映射，那么 '得分' 不能做特征。
# # 我们假设 'quality_label' 是一个独立的、需要预测的作文质量等级。

# X = df[feature_columns]
# y_text = df['quality_label']

# 2. 数据预处理
# 2.1 缺失值处理 (示例：简单填充，更优方法需视情况而定)
# X = X.fillna(X.mean())

# 2.2 标签编码
# encoder = LabelEncoder()
# y = encoder.fit_transform(y_text) # y 现在是数值型标签

# 2.3 特征标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

# 3. (选做) 新特征提取 - 此处略，但如果做了，X_scaled_df 需要包含这些新特征

# 4. 模型训练与评估
# 4.1 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y) # stratify=y 保持类别比例

# 4.2 选择模型并训练 (以随机森林为例)
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# 4.4 在测试集上预测并评估
# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_)) 
# # target_names 可以显示原始文本标签

# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

# 5. 结果分析与解释
# 5.3 特征重要性 (对于基于树的模型如随机森林)
# importances = model.feature_importances_
# feature_importance_df = pd.DataFrame({'feature': feature_columns, 'importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
# print("\nFeature Importances:\n", feature_importance_df)

# plt.figure(figsize=(10, 6))
# sns.barplot(x='importance', y='feature', data=feature_importance_df)
# plt.title('Feature Importances for Essay Quality Prediction')
# plt.show()

# --- 思考与讨论 ---
# 1. 关键因素解读：根据特征重要性，哪些因素对作文质量影响最大？这与教育学理论或写作教学经验是否一致？
# 2. 模型局限性：当前模型的准确率如何？在哪些类别的作文上表现较差？可能的原因是什么？
# 3. 如何应用：这个模型可以如何帮助教师（例如，辅助筛选需要重点关注的作文）或学生（例如，提供写作建议）？
# 4. 选做部分：如果提取了新的文本特征，它们是否提升了模型性能？哪些文本特征比较重要？
```

**给学习者的重要提示：**
* **明确标签：** 首先要非常清楚作文质量标签 `quality_label` 具体有哪些类别，以及它们是如何定义的。这会影响模型的选择和评估。
* **特征 `得分` 的处理：** 如果 `得分` 这个特征是人工给出的综合评分，并且 `quality_label` 是根据这个 `得分` 划分出来的（例如，90分以上是“优秀”，80-89是“良好”等），那么在预测 `quality_label` 时，**不应该直接使用 `得分` 作为模型的输入特征**，因为这会导致信息泄露和模型过于乐观。你需要确认 `得分` 是否是独立的原始特征，还是与 `quality_label` 高度相关甚至同源。如果同源，应将其排除在输入特征之外，或者将任务转化为直接预测 `得分` （回归问题）。本习题明确是“分类”，所以要预测的是等级标签。
* **类别不平衡：** 如果不同质量等级的作文数量差异很大（例如，“优秀”的很少，“合格”的很多），需要注意类别不平衡对模型训练和评估的影响。可能需要采用过采样、欠采样或使用对类别不平衡不敏感的评估指标（如F1分数、AUC-PR）或算法。
* **迭代优化：** 模型构建是一个不断尝试和优化的过程。可以尝试不同的算法、调整参数、做更细致的特征工程来提升模型性能。
