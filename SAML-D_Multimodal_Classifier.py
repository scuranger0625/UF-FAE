import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, concat_ws, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, LinearSVC
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# === 1. 建立 SparkSession（單一入口） ===
spark = SparkSession.builder.appName("SAML-D Multimodal ML with TimeSeries Split").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")  # 時間格式向下相容，避免 datetime 解析錯誤

# === 2. 讀取多模態特徵融合的 parquet（已經有圖特徵與原始資料） ===
df = spark.read.parquet("gs://saml-d/SAML-D_with_graph_centrality.parquet")

# === 3. 合併 Date+Time 成 timestamp，轉成 long 格式可排序 ===
df = df.withColumn(
    "timestamp",
    unix_timestamp(concat_ws(" ", col("Date"), col("Time")), "yyyy-MM-dd HH:mm:ss").cast("long")
)

# === 4. 嚴格依時間序列切分（按 timestamp 升冪排序，80%訓練 20%測試）===
df = df.orderBy("timestamp")      # 保證資料是依交易時間遞增
total = df.count()                # 資料總筆數
split_idx = int(total * 0.8)      # 切割點，80%給訓練集
train_data = df.limit(split_idx)  # 取前 split_idx 筆為訓練集
test_data = df.subtract(train_data)  # 剩下 20% 為測試集（保證嚴格未來資訊不可見）

# === 5. 特徵工程設計：分類特徵 + 數值特徵 ===
categorical_cols = [
    "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location", "Payment_type"
]
numeric_cols = [
    "Amount",  # 交易金額
    # 下方皆為圖結構 derived features
    "group_node_count", "group_edge_count", "group_bidirect_ratio",
    "sender_degree", "receiver_degree",
    "sender_closeness", "receiver_closeness",
    "sender_betweenness", "receiver_betweenness"
]

# === 6. 轉換類別型特徵為數字向量（先編碼再 OneHot）===
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in categorical_cols]

# 數值+OneHot編碼合併為 features 向量
feature_cols = numeric_cols + [f"{c}_vec" for c in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# === 7. 定義多個經典 ML 模型 ===
models = {
    "Logistic Regression": LogisticRegression(labelCol="Is_laundering", featuresCol="features"),
    "Decision Tree": DecisionTreeClassifier(labelCol="Is_laundering", featuresCol="features"),
    "Random Forest": RandomForestClassifier(labelCol="Is_laundering", featuresCol="features", numTrees=100),
    "SVM (LinearSVC)": LinearSVC(labelCol="Is_laundering", featuresCol="features")
}

# === 8. 設定各種常用分類評估指標（AUC、Precision、Recall、F1）===
binary_eval = BinaryClassificationEvaluator(labelCol="Is_laundering", metricName="areaUnderROC")
precision_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="precisionByLabel")
recall_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="recallByLabel")
f1_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="f1")

# === 9. 依序訓練每個模型，輸出訓練時間、各指標、特徵重要性（Top 10） ===
for name, clf in models.items():
    print(f"🔹 {name}")
    start = time.time()
    pipeline = Pipeline(stages=indexers + encoders + [assembler, clf])
    model = pipeline.fit(train_data)       # 模型訓練（僅用過去資料）
    predictions = model.transform(test_data)  # 測試集預測（未來資料）
    elapsed = time.time() - start

    # === 9.1 輸出各指標 ===
    auc = binary_eval.evaluate(predictions)
    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    print(f"   🕒 訓練+預測時間：{elapsed:.2f} 秒")
    print(f"   📈 AUC          ：{auc:.4f}")
    print(f"   🎯 Precision    ：{precision:.4f}")
    print(f"   🎯 Recall       ：{recall:.4f}")
    print(f"   🧮 F1 Score     ：{f1:.4f}\n")

    # === 9.2 特徵重要性（僅 DT/RF 有 .featureImportances）===
    if name in ["Decision Tree", "Random Forest"]:
        importances = model.stages[-1].featureImportances
        print("   🔬 Feature Importances:")
        feature_names = numeric_cols.copy()
        # 根據類別數量決定 OneHot 輸出長度（簡化設 20，實務可用字典動態查）
        for c in categorical_cols:
            ohe_size = 20
            feature_names.extend([f"{c}_vec_{i}" for i in range(ohe_size)])
        imp_list = list(importances)
        sorted_features = sorted(zip(feature_names, imp_list), key=lambda x: -x[1])[:10]
        for fname, score in sorted_features:
            print(f"      {fname:<25} {score:.4f}")
        print()
    # === 9.3 線性模型（Logistic/SVM）則印前10大絕對係數 ===
    elif name == "Logistic Regression":
        coefs = model.stages[-1].coefficients.toArray()
        top_idx = abs(coefs).argsort()[-10:][::-1]
        for idx in top_idx:
            print(f"      feature_{idx:<2} abs(coef): {abs(coefs[idx]):.4f}")
        print()
    elif name == "SVM (LinearSVC)":
        coefs = model.stages[-1].coefficients.toArray()
        top_idx = abs(coefs).argsort()[-10:][::-1]
        for idx in top_idx:
            print(f"      feature_{idx:<2} abs(coef): {abs(coefs[idx]):.4f}")
        print()

print("✅ 嚴格時間序列驗證與多模態特徵融合 ML pipeline 執行完畢！")
