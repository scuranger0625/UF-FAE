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

# === 1. 建立 SparkSession ===
spark = SparkSession.builder.appName("SAML-D Baseline ML with TimeSeries Split").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")  # 避免 datetime 解析錯誤

# === 2. 讀取 Parquet 原始資料 ===
df = spark.read.parquet("gs://saml-d/SAML-D.parquet")  # 注意！這裡不要用_with_graph_centrality

# === 3. 合併日期/時間（做成 timestamp，之後用於時間序列切分）===
df = df.withColumn(
    "timestamp",
    unix_timestamp(concat_ws(" ", col("Date"), col("Time")), "yyyy-MM-dd HH:mm:ss").cast("long")
)

# === 4. 嚴格時間序列切分（按時間排序、80%訓練/20%測試）===
df = df.orderBy("timestamp")
total = df.count()
split_idx = int(total * 0.8)
train_data = df.limit(split_idx)
test_data = df.subtract(train_data)

# === 5. 僅用原始 SAML-D 欄位（不含任何圖特徵）===
categorical_cols = [
    "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location", "Payment_type"
]
numeric_cols = [
    "Amount"  # 只用金額
    # 注意：不包含圖結構特徵
]

# === 6. 類別型轉數值（StringIndexer + OneHotEncoder）===
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in categorical_cols]
feature_cols = numeric_cols + [f"{c}_vec" for c in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# === 7. 定義 ML 模型 ===
models = {
    "Logistic Regression": LogisticRegression(labelCol="Is_laundering", featuresCol="features"),
    "Decision Tree": DecisionTreeClassifier(labelCol="Is_laundering", featuresCol="features"),
    "Random Forest": RandomForestClassifier(labelCol="Is_laundering", featuresCol="features", numTrees=100),
    "SVM (LinearSVC)": LinearSVC(labelCol="Is_laundering", featuresCol="features")
}

# === 8. 評估指標 ===
binary_eval = BinaryClassificationEvaluator(labelCol="Is_laundering", metricName="areaUnderROC")
precision_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="precisionByLabel")
recall_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="recallByLabel")
f1_eval = MulticlassClassificationEvaluator(labelCol="Is_laundering", predictionCol="prediction", metricName="f1")

# === 9. 訓練每個模型，並印出評估指標與特徵重要性 ===
for name, clf in models.items():
    print(f"🔹 {name}")
    start = time.time()
    pipeline = Pipeline(stages=indexers + encoders + [assembler, clf])
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

    # === 特徵重要性（只有DT/RF有 .featureImportances）===
    if name in ["Decision Tree", "Random Forest"]:
        importances = model.stages[-1].featureImportances
        print("   🔬 Feature Importances:")
        feature_names = numeric_cols.copy()
        for c in categorical_cols:
            ohe_size = 20
            feature_names.extend([f"{c}_vec_{i}" for i in range(ohe_size)])
        imp_list = list(importances)
        sorted_features = sorted(zip(feature_names, imp_list), key=lambda x: -x[1])[:10]
        for fname, score in sorted_features:
            print(f"      {fname:<25} {score:.4f}")
        print()
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

print("✅ 嚴格時間序列驗證（只用原始特徵）ML pipeline 執行完畢！")
