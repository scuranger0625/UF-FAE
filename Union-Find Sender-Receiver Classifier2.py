import polars as pl
from collections import defaultdict
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

# === 1. 讀取 CSV 檔案 ===
file_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.csv"
df = pl.read_csv(file_path, infer_schema_length=0)

# === 2. 取出邊列表（Sender, Receiver）===
edges = df.select(["Sender_account", "Receiver_account"]).rows()

# === 3. Union-Find 分群 ===
parent = {}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x]) # ← 路徑壓縮：將 x 直接指向根節點
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

# 初始化 parent
accounts = set()
for u, v in edges:
    accounts.update([u, v])
for acc in accounts:
    parent[acc] = acc
for u, v in edges:
    union(u, v)

# 建立群組
groups = defaultdict(list)
for acc in accounts:
    root = find(acc)
    groups[root].append(acc)

print(f"✅ 總共分成 {len(groups)} 個子圖群組")

# === 4. 統計出入度 ===
in_df = (
    df.select(["Receiver_account"])
      .group_by("Receiver_account")
      .agg(pl.len().alias("InDegree"))
      .rename({"Receiver_account": "Account"})
)

out_df = (
    df.select(["Sender_account"])
      .group_by("Sender_account")
      .agg(pl.len().alias("OutDegree"))
      .rename({"Sender_account": "Account"})
)

degree_df = in_df.join(out_df, on="Account", how="outer").fill_null(0)

# 轉為查詢用字典
in_degree = dict(zip(degree_df["Account"], degree_df["InDegree"]))
out_degree = dict(zip(degree_df["Account"], degree_df["OutDegree"]))

# === 5. 每個群組的 Sender / Receiver 統計與列印 ===
for idx, (gid, nodes) in enumerate(groups.items(), 1):
    sender_set = [n for n in nodes if in_degree.get(n, 0) == 0]
    receiver_set = [n for n in nodes if out_degree.get(n, 0) == 0]
    print(f"群組 {idx}: 節點數 = {len(nodes)}，Sender = {len(sender_set)}，Receiver = {len(receiver_set)}")

indexed_group_roots = list(groups.keys()) # ⬅️ 一定要有這行！

# 先建立群組編號對應的節點數資料：[(編號, 節點數)]
group_node_counts = []

for idx, root in enumerate(indexed_group_roots, 1):  # idx 從 1 開始
    node_count = len(groups[root])
    group_node_counts.append((idx, node_count))

# 依節點數排序（由多到少）
group_node_counts.sort(key=lambda x: x[1], reverse=True)

# 使用者輸入前 n 名（含並列）
n = int(input("請輸入要列出節點數最多的前幾名群組（含並列）："))

# 取得第 n 名的節點數門檻
threshold = group_node_counts[n - 1][1]

# 篩出符合條件的群組（含並列）
top_groups = [(idx, count) for idx, count in group_node_counts if count >= threshold]
top_group_indices = set(idx for idx, _ in top_groups)

# 統計每個群組的洗錢筆數
laundering_per_group = {idx: 0 for idx in top_group_indices}

for row in df.iter_rows(named=True):
    if int(row["Is_laundering"]) == 1:
        sender = row["Sender_account"]
        if sender in parent:
            root = find(sender)
            if root in indexed_group_roots:
                idx = indexed_group_roots.index(root) + 1
                if idx in laundering_per_group:
                    laundering_per_group[idx] += 1

# 顯示結果
print(f"\n📊 節點數最多的前 {n} 名（含並列）群組：")
for idx, count in top_groups:
    laundering_count = laundering_per_group.get(idx, 0)
    if laundering_count > 0:
        print(f"🔹 群組 {idx}：節點數 = {count}，洗錢交易 = {laundering_count} 筆")
    else:
        print(f"⚪ 群組 {idx}：節點數 = {count}，🚫 無洗錢交易")

# 統計完全沒有洗錢的群組數
no_laundering = sum(1 for v in laundering_per_group.values() if v == 0)

print(f"\n共 {len(top_groups)} 個群組（含並列）符合條件")
print(f"🔴 偵測到 {sum(laundering_per_group.values())} 筆洗錢交易")
print(f"✅ 其中 {no_laundering} 個群組完全沒有洗錢交易")


# 輸入群組編號（例如：1~n）
group_number = int(input("請輸入群組編號："))
group_root = indexed_group_roots[group_number - 1]
group_accounts = set(groups[group_root])

# 建立有向圖
G = nx.DiGraph()
laundering_times = []  # 存放洗錢邊的 datetime 時間

# 篩選該群組內的交易邊
for row in df.iter_rows(named=True):
    sender = row["Sender_account"]
    receiver = row["Receiver_account"]
    laundering = int(row["Is_laundering"])
    if sender in group_accounts and receiver in group_accounts:
        G.add_edge(sender, receiver, laundering=laundering)
        if laundering == 1:
            # 將 Date 和 Time 合併後轉為 datetime 物件
            dt_str = f"{row['Date']} {row['Time']}"  # 例如 '2022-10-07 10:35:19'
            try:
                laundering_times.append(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass  # 無法解析則略過

# 分類邊
red_edges = [(u, v) for u, v, d in G.edges(data=True) if d["laundering"] == 1]
gray_edges = [(u, v) for u, v, d in G.edges(data=True) if d["laundering"] == 0]
node_count = G.number_of_nodes()

# 顯示統計資訊
print(f"\n📊 群組 {group_number} 的交易統計：")
print(f"👥 帳戶總數（節點數）：{node_count}")
print(f"➡️ 總交易邊數：{G.number_of_edges()}")
print(f"🔴 洗錢交易邊數（紅色）：{len(red_edges)}")
print(f"⚪ 正常交易邊數（灰色）：{len(gray_edges)}")

# 顯示洗錢時間範圍
if laundering_times:
    print(f"🕒 最早洗錢時間：{min(laundering_times)}")
    print(f"🕒 最晚洗錢時間：{max(laundering_times)}")
else:
    print("⚠️ 此群組沒有洗錢交易，因此無時間範圍可顯示。")

# -------- 圖一：原始交易圖（紅 + 灰）--------
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=300, node_color="#a3c1da")
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, edgelist=gray_edges, edge_color="gray", arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color="red", arrows=True, width=2)
plt.title(f"Group {group_number} Full Transaction Graph\n(Red Edges = Money Laundering)", fontsize=14)
plt.axis("off")
plt.show()

# -------- 圖二：僅洗錢子圖 --------
G_red = nx.DiGraph()
G_red.add_edges_from(red_edges)

pos_red = nx.spring_layout(G_red, seed=42)
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G_red, pos_red, node_size=300, node_color="#ffb3b3")
nx.draw_networkx_labels(G_red, pos_red, font_size=8)
nx.draw_networkx_edges(G_red, pos_red, edge_color="red", arrows=True, width=2)
plt.title(f"Group {group_number} Money Laundering Subgraph\n(Only Red Edges)", fontsize=14)
plt.axis("off")
plt.show()

# 建立 root 編號與對應 index 的 mapping（和圖示時一致）
indexed_group_roots = list(groups.keys())

# 建立群組洗錢筆數的計數 dict：{群組編號: 洗錢筆數}
laundering_count = {}

# 檢查每一筆交易是否在某個群組中且標記為洗錢
for row in df.iter_rows(named=True):
    if int(row["Is_laundering"]) == 1:
        sender = row["Sender_account"]
        receiver = row["Receiver_account"]

        # 找出 sender 所屬群組 root
        if sender in parent:
            root = find(sender)
            if root in indexed_group_roots:
                idx = indexed_group_roots.index(root) + 1  # 群組編號從 1 開始
                laundering_count[idx] = laundering_count.get(idx, 0) + 1

# 排序後輸出結果
sorted_groups = sorted(laundering_count.items(), key=lambda x: x[1], reverse=True)

print("📊 含洗錢交易的群組（依洗錢筆數排序）：")
for gid, count in sorted_groups:
    print(f"✅ 群組 {gid}：有 {count} 筆洗錢交易")

print(f"\n🔴 總共有 {len(laundering_count)} 個群組包含洗錢交易")
