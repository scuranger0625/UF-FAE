import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, concat_ws, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, LinearSVC
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# === 1. 建立 SparkSession（單一入口） ===
spark = SparkSession.builder.appName("SAML-D GraphOnly ML with TimeSeries Split").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")  # 避免 datetime 解析錯誤

# === 2. 讀取多模態特徵 parquet（有圖特徵） ===
df = spark.read.parquet("gs://saml-d/SAML-D_with_graph_centrality.parquet")

# === 3. 合併 Date+Time 成 timestamp（照舊，切時間用） ===
df = df.withColumn(
    "timestamp",
    unix_timestamp(concat_ws(" ", col("Date"), col("Time")), "yyyy-MM-dd HH:mm:ss").cast("long")
)

# === 4. 嚴格時間序列切分（升冪排序，80%訓練、20%測試）===
df = df.orderBy("timestamp")
total = df.count()
split_idx = int(total * 0.8)
train_data = df.limit(split_idx)
test_data = df.subtract(train_data)

# === 5. 只取圖特徵作為 numeric features ===
graph_cols = [
    "group_node_count", "group_edge_count", "group_bidirect_ratio",
    "sender_degree", "receiver_degree",
    "sender_closeness", "receiver_closeness",
    "sender_betweenness", "receiver_betweenness"
]
# 【不加入任何原始交易、帳戶、金額、地理等欄位】

assembler = VectorAssembler(inputCols=graph_cols, outputCol="features")

# === 6. 定義 ML 模型 ===
models = {
    "Logistic Regression": LogisticRegression(labelCol="Is_laundering", featuresCol="features"),
    "Decision Tree": DecisionTreeClassifier(labelCol="Is_laundering", featuresCol="features"),
    "Random Forest": RandomForestClassifier(labelCol="Is_laundering", featuresCol="features", numTrees=100),
    "SVM (LinearSVC)": LinearSVC(labelCol="Is_laundering", featuresCol="features")
}

# === 7. 設定評估指標 ===
binary_eval = BinaryClassificationEvaluator(labelCol="Is_laundering", metricName="areaUnderROC")
precision_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="precisionByLabel")
recall_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="recallByLabel")
f1_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="f1")

# === 8. 執行訓練/預測/重要性分析 ===
for name, clf in models.items():
    print(f"🔹 {name} (Graph-Only Features)")
    start = time.time()
    pipeline = Pipeline(stages=[assembler, clf])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    elapsed = time.time() - start

    auc = binary_eval.evaluate(predictions)
    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)

    print(f"   🕒 訓練+預測時間：{elapsed:.2f} 秒")
    print(f"   📈 AUC          ：{auc:.4f}")
    print(f"   🎯 Precision    ：{precision:.4f}")
    print(f"   🎯 Recall       ：{recall:.4f}")
    print(f"   🧮 F1 Score     ：{f1:.4f}\n")

    # === 特徵重要性分析 ===
    if name in ["Decision Tree", "Random Forest"]:
        importances = model.stages[-1].featureImportances
        feature_names = graph_cols
        imp_list = list(importances)
        sorted_features = sorted(zip(feature_names, imp_list), key=lambda x: -x[1])[:8]
        for fname, score in sorted_features:
            print(f"      {fname:<25} {score:.4f}")
        print()
    elif name == "Logistic Regression":
        coefs = model.stages[-1].coefficients.toArray()
        top_idx = abs(coefs).argsort()[-8:][::-1]
        for idx in top_idx:
            print(f"      {graph_cols[idx]:<25} abs(coef): {abs(coefs[idx]):.4f}")
        print()
    elif name == "SVM (LinearSVC)":
        coefs = model.stages[-1].coefficients.toArray()
        top_idx = abs(coefs).argsort()[-8:][::-1]
        for idx in top_idx:
            print(f"      {graph_cols[idx]:<25} abs(coef): {abs(coefs[idx]):.4f}")
        print()

print("✅ 嚴格時間序列驗證（僅用圖特徵）ML pipeline 執行完畢！")
