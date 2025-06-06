# This module implements a Graph Attention Network (GAT) pipeline for student classroom allocation.
# It includes functions for computing group scores, assigning classrooms, and exporting results
# to both Neo4j and D3.js visualization formats.

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import json
from neo4j import GraphDatabase
from torch_geometric.data import Data
from your_module import DualHeadGAT  # make sure to import your GAT model

# Mapping between relationship types and their numeric indices
EDGE_TYPE_MAP = {"Friend": 0, "Neutral": 1, "Conflict": 2}
EDGE_LABELS = {0: "Friend", 1: "Neutral", 2: "Conflict"}

def compute_group_score(indices, preds, distress, edge_index, edge_types, sliders):
    """
    Compute a score for a group of students based on various factors.
    
    The score considers:
    - Academic performance variance (GPA standard deviation)
    - Wellbeing variance (distress standard deviation)
    - Number of friendships within the group
    - Number of conflicts within the group
    
    Args:
        indices (list): Indices of students in the group
        preds (Tensor): Predicted academic performance scores
        distress (Tensor): Student wellbeing/distress scores
        edge_index (Tensor): Graph edge connections
        edge_types (Tensor): Types of relationships between students
        sliders (dict): Weighting factors for different aspects
    
    Returns:
        float: Combined score for the group (lower is better)
    """
    # Calculate standard deviations for academic performance and wellbeing
    std_gpa = torch.std(preds[indices])
    std_wellbeing = torch.std(distress[indices])
    friend_score = 0
    conflict_score = 0

    # Count friendships and conflicts within the group
    for i, (src, tgt) in enumerate(edge_index.t()):
        if src.item() in indices and tgt.item() in indices:
            label = edge_types[i].item()
            if label == EDGE_TYPE_MAP["Friend"]:
                friend_score -= 1  # Negative because lower score is better
            elif label == EDGE_TYPE_MAP["Conflict"]:
                conflict_score += 1

    # Combine scores using slider weights
    total = (
        sliders["academic_balance"] * std_gpa.item() +
        sliders["wellbeing"] * std_wellbeing.item() +
        sliders["friendship_retention"] * friend_score +
        sliders["conflict_penalty"] * conflict_score
    )
    return total

def assign_classrooms(embeddings, preds, edge_index, edge_types, distress, sliders, class_size=40):
    """
    Assign students to classrooms using an iterative optimization approach.
    
    The function:
    1. Makes multiple attempts with random initializations
    2. For each attempt, creates groups of specified size
    3. Computes group scores considering academic performance, wellbeing, friendships, and conflicts
    4. Returns the assignment with the best overall score
    
    Args:
        embeddings (Tensor): Student embeddings from GAT
        preds (Tensor): Predicted academic performance scores
        edge_index (Tensor): Graph edge connections
        edge_types (Tensor): Types of relationships between students
        distress (Tensor): Student wellbeing/distress scores
        sliders (dict): Weighting factors for different aspects
        class_size (int): Target size for each classroom
    
    Returns:
        list: Best classroom assignments for each student
    """
    num_students = len(preds)
    best_score = float("inf")
    best_labels = None
    indices = list(range(num_students))

    # Try multiple random initializations
    for attempt in range(10):
        random.shuffle(indices)
        labels = [0] * num_students
        i = 0
        cid = 0
        
        # Create groups of specified size
        while i < num_students:
            group = indices[i:i+class_size]
            for idx in group:
                labels[idx] = cid
            i += class_size
            cid += 1

        # Compute scores for each group
        group_scores = []
        for g in range(cid):
            group_idxs = [j for j in range(num_students) if labels[j] == g]
            group_score = compute_group_score(group_idxs, preds, distress, edge_index, edge_types, sliders)
            group_scores.append(group_score)

        # Update best assignment if current is better
        total_score = sum(group_scores)
        if total_score < best_score:
            best_score = total_score
            best_labels = labels

    return best_labels

def run_gat_and_export(student_df, features, labels, edge_index, edge_types, sliders, neo4j_config, d3_path):
    """
    Main pipeline function that runs the GAT model and exports results.
    
    The pipeline:
    1. Initializes and trains the DualHeadGAT model
    2. Generates classroom assignments
    3. Exports results to D3.js and Neo4j
    
    Args:
        student_df (DataFrame): Student data
        features (Tensor): Student feature vectors
        labels (Tensor): Target values for training
        edge_index (Tensor): Graph edge connections
        edge_types (Tensor): Types of relationships between students
        sliders (dict): Weighting factors for different aspects
        neo4j_config (dict): Neo4j connection configuration
        d3_path (str): Path to save D3.js visualization data
    
    Returns:
        None: Results are exported to files and database
    """
    print("🎓 Initializing DualHeadGAT model...")
    model = DualHeadGAT(features.shape[1], hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_node = nn.MSELoss()
    loss_edge = nn.CrossEntropyLoss()

    # Train the GAT model
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        node_preds, edge_preds, embeddings, _ = model(features, edge_index)
        loss = loss_node(node_preds, labels) + loss_edge(edge_preds, edge_types)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Generate classroom assignments
    print("📊 Allocating students into classrooms based on slider weights...")
    model.eval()
    with torch.no_grad():
        node_preds, edge_preds, embeddings, attn_weights = model(features, edge_index)
        distress = features[:, -1]  # assuming distress is last column
        class_labels = assign_classrooms(embeddings, node_preds, edge_index, edge_types, distress, sliders)

    # Prepare node data for visualization
    nodes = [{
        "id": str(i),
        "cluster": class_labels[i],
        "score": float(node_preds[i].item()),
        "embedding": embeddings[i].tolist()
    } for i in range(len(features))]

    # Prepare edge data for visualization
    edges = []
    for i, (src, tgt) in enumerate(edge_index.t().tolist()):
        pred = edge_preds[i]
        edge_type = EDGE_LABELS[pred.argmax().item()]
        confidence = torch.softmax(pred, dim=0).max().item()
        attention = round(attn_weights[i], 4)
        edges.append({
            "source": str(src),
            "target": str(tgt),
            "type": edge_type,
            "confidence": round(confidence, 4),
            "attention": attention
        })

    # Export to D3.js JSON format
    with open(d3_path, "w") as f:
        json.dump({"nodes": nodes, "links": edges}, f, indent=2)

    # Export to Neo4j
    print("🌐 Exporting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"]))
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create student nodes
        for node in nodes:
            session.run("""
                CREATE (s:Student {id: $id, cluster: $cluster, score: $score, embedding: $embedding})
            """, node)
        
        # Create relationship edges
        for edge in edges:
            session.run("""
                MATCH (a:Student {id: $source}), (b:Student {id: $target})
                CREATE (a)-[:RELATES {type: $type, confidence: $confidence, attention: $attention}]->(b)
            """, edge)

    print("✅ GAT training, allocation, and export complete.")























# # gat_pipeline.py

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from sklearn.cluster import KMeans
# from neo4j import GraphDatabase
# import json
# import os

# # -------------------------
# # GAT MODEL DEFINITIONS
# # -------------------------
# class DualHeadGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_node_dim, out_edge_dim, heads=1):
#         super().__init__()
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
#         self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)

#         self.node_predictor = torch.nn.Linear(hidden_channels * heads, out_node_dim)  # exam score
#         self.edge_predictor = torch.nn.Bilinear(hidden_channels * heads, hidden_channels * heads, out_edge_dim)  # edge type

#     def forward(self, x, edge_index):
#         x = F.elu(self.gat1(x, edge_index))
#         x = F.elu(self.gat2(x, edge_index))

#         node_preds = self.node_predictor(x)  # shape: [num_nodes, 1]

#         edge_preds = []
#         for src, tgt in edge_index.t():
#             edge_pred = self.edge_predictor(x[src], x[tgt])  # shape: [out_edge_dim]
#             edge_preds.append(edge_pred)
#         edge_preds = torch.stack(edge_preds)

#         return node_preds, edge_preds, x  # x is embedding


# # -------------------------
# # EDGE TYPE MAPPER
# # -------------------------
# EDGE_LABELS = ["Friend", "Neutral", "Conflict"]

# def classify_edge_type(pred_tensor):
#     return EDGE_LABELS[pred_tensor.argmax().item()]


# # -------------------------
# # GAT RUNNER + EXPORT
# # -------------------------
# def run_gat_and_export(data, neo4j_config, d3_path):
#     print("🔄 Initializing DualHeadGAT model...")
#     model = DualHeadGAT(
#         in_channels=data.x.shape[1],
#         hidden_channels=32,
#         out_node_dim=1,
#         out_edge_dim=3
#     )

#     model.eval()
#     with torch.no_grad():
#         print("🔍 Running forward pass...")
#         node_preds, edge_preds, embeddings = model(data.x, data.edge_index)

#     print("📊 Performing KMeans clustering...")
#     kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings.numpy())
#     cluster_labels = kmeans.labels_

#     print("📦 Formatting node and edge data...")
#     nodes = []
#     for idx, embedding in enumerate(embeddings):
#         nodes.append({
#             "id": str(idx),
#             "cluster": int(cluster_labels[idx]),
#             "score": float(node_preds[idx].item()),
#             "embedding": embedding.tolist()
#         })

#     edges = []
#     for i, (src, tgt) in enumerate(data.edge_index.t().tolist()):
#         edge_type = classify_edge_type(edge_preds[i])
#         edges.append({"source": str(src), "target": str(tgt), "type": edge_type})

#     print("💾 Saving to D3 JSON file...")
#     with open(d3_path, "w") as f:
#         json.dump({"nodes": nodes, "links": edges}, f, indent=2)

#     print("🌐 Connecting to Neo4j...")
#     driver = GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"]))
#     with driver.session() as session:
#         print("🧹 Clearing existing graph...")
#         session.run("MATCH (n) DETACH DELETE n")

#         print("🔁 Uploading nodes to Neo4j...")
#         for node in nodes:
#             session.run("""
#                 CREATE (s:Student {id: $id, cluster: $cluster, score: $score, embedding: $embedding})
#             """, node)

#         print("🔁 Uploading edges to Neo4j...")
#         for edge in edges:
#             session.run("""
#                 MATCH (a:Student {id: $source}), (b:Student {id: $target})
#                 CREATE (a)-[:RELATES {type: $type}]->(b)
#             """, edge)

#     print("✅ GAT embeddings, clusters, and edges exported to Neo4j and JSON.")


# # # gat_pipeline.py

# # import torch
# # import torch.nn.functional as F
# # from torch_geometric.nn import GATConv
# # from sklearn.cluster import KMeans
# # from neo4j import GraphDatabase
# # import json
# # import os

# # # -------------------------
# # # GAT MODEL DEFINITIONS
# # # -------------------------
# # class DualHeadGAT(torch.nn.Module):
# #     def __init__(self, in_channels, hidden_channels, out_node_dim, out_edge_dim, heads=1):
# #         super().__init__()
# #         self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
# #         self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)

# #         self.node_predictor = torch.nn.Linear(hidden_channels * heads, out_node_dim)  # exam score
# #         self.edge_predictor = torch.nn.Bilinear(hidden_channels * heads, hidden_channels * heads, out_edge_dim)  # edge type

# #     def forward(self, x, edge_index):
# #         x = F.elu(self.gat1(x, edge_index))
# #         x = F.elu(self.gat2(x, edge_index))

# #         node_preds = self.node_predictor(x)  # shape: [num_nodes, 1]

# #         edge_preds = []
# #         for src, tgt in edge_index.t():
# #             edge_pred = self.edge_predictor(x[src], x[tgt])  # shape: [out_edge_dim]
# #             edge_preds.append(edge_pred)
# #         edge_preds = torch.stack(edge_preds)

# #         return node_preds, edge_preds, x  # x is embedding


# # # -------------------------
# # # EDGE TYPE MAPPER
# # # -------------------------
# # EDGE_LABELS = ["Friend", "Neutral", "Conflict"]

# # def classify_edge_type(pred_tensor):
# #     return EDGE_LABELS[pred_tensor.argmax().item()]


# # # -------------------------
# # # GAT RUNNER + EXPORT
# # # -------------------------
# # def run_gat_and_export(data, neo4j_config, d3_path):
# #     model = DualHeadGAT(
# #         in_channels=data.x.shape[1],
# #         hidden_channels=32,
# #         out_node_dim=1,
# #         out_edge_dim=3
# #     )

# #     model.eval()
# #     with torch.no_grad():
# #         node_preds, edge_preds, embeddings = model(data.x, data.edge_index)

# #     # Cluster embeddings for classroom grouping
# #     kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings.numpy())
# #     cluster_labels = kmeans.labels_

# #     # Format for Neo4j and D3
# #     nodes = []
# #     for idx, embedding in enumerate(embeddings):
# #         nodes.append({
# #             "id": str(idx),
# #             "cluster": int(cluster_labels[idx]),
# #             "score": float(node_preds[idx].item()),
# #             "embedding": embedding.tolist()
# #         })

# #     edges = []
# #     for i, (src, tgt) in enumerate(data.edge_index.t().tolist()):
# #         edge_type = classify_edge_type(edge_preds[i])
# #         edges.append({"source": str(src), "target": str(tgt), "type": edge_type})

# #     # Export to D3 JSON
# #     with open(d3_path, "w") as f:
# #         json.dump({"nodes": nodes, "links": edges}, f, indent=2)

# #     # Export to Neo4j
# #     driver = GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"]))
# #     with driver.session() as session:
# #         session.run("MATCH (n) DETACH DELETE n")
# #         for node in nodes:
# #             session.run("""
# #                 CREATE (s:Student {id: $id, cluster: $cluster, score: $score, embedding: $embedding})
# #             """, node)

# #         for edge in edges:
# #             session.run("""
# #                 MATCH (a:Student {id: $source}), (b:Student {id: $target})
# #                 CREATE (a)-[:RELATES {type: $type}]->(b)
# #             """, edge)

# #     print("✅ GAT embeddings, clusters, and edges exported to Neo4j and JSON.")
