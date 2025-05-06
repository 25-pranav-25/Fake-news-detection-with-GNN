# 📰 Fake News Propagation Analysis Using Graph Neural Networks

This project explores the use of **Graph Neural Networks (GNNs)** to detect fake news on social media platforms by analyzing how information propagates through a network of users. Instead of relying solely on the content of news articles, we leverage the **structure of their spread** (retweet behavior) using graph-based deep learning models.

---

## 📂 Dataset: FakeNewsNet (UPFD - Politifact)

We use the [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset, integrated into [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) as the **UPFD** dataset.

- **Subset Used**: `Politifact`
- **Graphs**: 314 propagation trees (157 fake, 157 real)
- **Structure**:
  - Root node: News article
  - Leaf nodes: Twitter users who retweeted
  - Edges: Retweet relations
- **Node Features**:
  - 10-d user profile vector
  - 768-d BERT embedding of past tweets
  - Final feature vector: **778 dimensions**

---

## ⚙️ Preprocessing

- Converted all graphs from **directed to undirected**
- Loaded and combined **profile attributes** with **BERT-based tweet embeddings**
- Handled via built-in `UPFD` dataset loader in PyG

---

## 🧠 Model Architectures

### 🔷 Graph Convolutional Network (GCN)

- Two-layer GCN
- No dropout (suitable for shallow propagation trees)
- Lightweight and scalable

### 🔷 GNN with Differential Pooling (GNN-DP)

- Uses **GraphSAGE + BatchNorm** layers
- Two parallel GNNs:
  - One for node embedding
  - One for cluster assignment
- Performs **Differentiable Pooling** twice:
  - Cluster sizes reduced: 500 → 100 → 20
- Final prediction: mean pooling + softmax classification

---

## 📊 Evaluation

| Model       | Accuracy | F1 Score |
|-------------|----------|----------|
| **GCN**     | 0.8371   | 0.8552   |
| **GNN-DP**  | 0.8054   | 0.8037   |

🔍 **Observation**: The GCN model outperformed GNN-DP on this dataset, suggesting that simpler models may be more effective for shallow propagation structures like those in Politifact.

---

## 💬 Discussion

With the rise of generative AI tools like **GPT-4**, the volume and realism of fake content have dramatically increased. Traditional fake news detection methods (fact-checking, NLP-based analysis) are often not scalable in real-time.

Our project highlights how **GNNs can detect fake news** by leveraging the **propagation patterns** of information in social networks. By analyzing *how* news spreads rather than *what* it contains, GNNs provide a promising direction in combating misinformation at scale.

---

## 🔧 Requirements
- Python 3.8+
- PyTorch
- PyTorch Geometric
