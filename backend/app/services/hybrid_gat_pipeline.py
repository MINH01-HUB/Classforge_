# This module implements a hybrid Graph Attention Network (GAT) pipeline for classroom allocation.
# It combines graph neural networks with traditional optimization to create balanced classroom assignments
# while considering academic performance, wellbeing, friendships, and behavioral factors.
# This file is used for the hybrid GAT pipeline. It is not used in the main app.
import torch
import numpy as np
import json
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from app.models.model import TransformedStudentData
from app.services.dual_head_gat import DualHeadGAT, train_dual_head_gat
from app.services.neo4j_service import neo4j_service
from app import db

def run_gat_and_export(sliders):
    """
    Main pipeline function that processes student data and generates classroom allocations using a hybrid GAT approach.
    
    The pipeline consists of the following steps:
    1. Load student data from the database
    2. Prepare node features and targets for the GAT model
    3. Build a synthetic graph structure using Stochastic Block Model (SBM)
    4. Generate random edge labels for training
    5. Train the DualHeadGAT model
    6. Score students using attention-aware logic
    7. Perform classroom allocation with conflict avoidance
    8. Export results to Neo4j
    9. Generate D3-compatible visualization data
    
    Args:
        sliders (dict): Dictionary containing slider values for different weighting factors:
            - academicBalance: Weight for academic performance (0-100)
            - wellbeingDistribution: Weight for wellbeing considerations (0-100)
            - friendshipRetention: Weight for maintaining friendships (0-100)
            - behavioralConsiderations: Weight for behavioral factors (0-100)
    
    Returns:
        dict: Summary of the allocation process including:
            - neo4j_status: Status of Neo4j database update
            - d3_path: Path to the generated D3 visualization file
            - num_students: Total number of students processed
            - num_edges: Number of relationships in the graph
            - class_summary: Statistics for each classroom
            - message: Status message
    """
    # === 1. Load students from DB ===
    students = TransformedStudentData.query.all()
    if not students:
        raise ValueError("❌ No transformed student data found in the database.")
    n = len(students)

    # === 2. Prepare node features and targets ===
    # Extract and normalize student features for the GAT model
    features, node_targets = [], []
    for s in students:
        features.append([
            s.encoded_gender,
            s.encoded_immigrant_status,
            s.ses,
            s.achievement,
            s.psychological_distress
        ])
        node_targets.append(s.achievement)

    # Normalize features to [0,1] range for better model performance
    features = np.array(features)
    features = MinMaxScaler().fit_transform(features)
    x = torch.tensor(features, dtype=torch.float)
    node_targets = torch.tensor(node_targets, dtype=torch.float)

    # === 3. Build synthetic SBM graph structure ===
    # Create a Stochastic Block Model to generate realistic social network structure
    k = 4  # number of blocks (social groups)
    # Probability matrix for connections between blocks
    P = np.array([[0.8, 0.2, 0.1, 0.1],
                  [0.2, 0.7, 0.2, 0.1],
                  [0.1, 0.2, 0.6, 0.3],
                  [0.1, 0.1, 0.3, 0.5]])

    # Assign students to blocks and generate connections
    labels = np.random.randint(0, k, size=n)
    blocks = [np.where(labels == i)[0] for i in range(k)]
    adj = np.zeros((n, n))
    for i in range(k):
        for j in range(k):
            for u in blocks[i]:
                for v in blocks[j]:
                    if u < v and np.random.rand() < P[i][j]:
                        adj[u, v] = 1
                        adj[v, u] = 1

    # Convert to NetworkX graph
    G = nx.from_numpy_array(adj)

    # Identify influential students (top 10% by degree centrality)
    centrality = nx.degree_centrality(G)
    top_n = int(0.1 * n)
    top_influencers = {node for node, _ in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]}

    # Add node attributes for PyTorch Geometric
    for i in G.nodes():
        G.nodes[i]['x'] = x[i]
        G.nodes[i]['is_influencer'] = i in top_influencers

    # Convert to PyG format
    data = from_networkx(G)
    data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])
    edge_index = data.edge_index

    # === 4. Random edge labels for classification training ===
    # Generate random relationship types (0: Friend, 1: Neutral, 2: Conflict)
    edge_labels = torch.randint(0, 3, (edge_index.size(1),))

    # === 5. Train DualHeadGAT ===
    # Initialize and train the model
    model = DualHeadGAT(in_channels=data.num_features, hidden_channels=16, out_node_dim=1, out_edge_dim=3)
    train_dual_head_gat(model, data, node_targets, edge_labels)

    # Generate predictions
    model.eval()
    with torch.no_grad():
        node_pred, edge_logits, x_emb, attention = model(data.x, data.edge_index)

    # === 6. Score students using attention-aware logic ===
    # Extract weights from slider inputs
    academic_w = sliders.get("academicBalance", 50) / 100
    wellbeing_w = sliders.get("wellbeingDistribution", 50) / 100
    friendship_w = sliders.get("friendshipRetention", 50) / 100
    behavior_w = sliders.get("behavioralConsiderations", 50) / 100

    # Calculate base scores considering academic performance and wellbeing
    distress_tensor = torch.tensor([s.psychological_distress for s in students])
    base_score = academic_w * node_pred - wellbeing_w * distress_tensor

    # Track friendships and conflicts
    friendships = defaultdict(set)
    conflicts = defaultdict(set)
    friend_score_map = torch.zeros(n)

    # Process relationships and update scores
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        label = torch.argmax(edge_logits[i]).item()
        attn = attention[i]
        if label == 0:  # Friendship
            friend_score_map[src] += attn * friendship_w
            friend_score_map[tgt] += attn * friendship_w
            friendships[src].add(tgt)
            friendships[tgt].add(src)
        elif label == 2:  # Conflict
            base_score[src] -= attn * behavior_w
            base_score[tgt] -= attn * behavior_w
            conflicts[src].add(tgt)
            conflicts[tgt].add(src)

    # Combine scores for final ranking
    scores = base_score + friend_score_map
    sorted_indices = torch.argsort(scores, descending=True)

    # === 7. Classroom allocation with conflict-avoidance + friend-aware sorting ===
    # Define class size constraints
    max_class_size = 40
    min_class_size = 35
    num_classes = max(1, n // min_class_size)
    class_sets = {f"Class_{i+1}": set() for i in range(num_classes)}
    class_buckets = {f"Class_{i+1}": [] for i in range(num_classes)}
    classroom_assignments = {}

    # Assign students to classes
    for idx in sorted_indices:
        sid = idx.item()
        student = students[sid]

        # Find best class considering conflicts and friendships
        best_class, max_friends = None, -1
        for cid, members in class_sets.items():
            if len(members) >= max_class_size:
                continue
            if any(m in conflicts[sid] for m in members):
                continue
            friend_count = sum(1 for m in members if m in friendships[sid])
            if friend_count > max_friends:
                best_class, max_friends = cid, friend_count

        # If no suitable class found, assign to smallest class
        if not best_class:
            best_class = min(class_sets.items(), key=lambda kv: len(kv[1]))[0]

        # Update assignments
        class_sets[best_class].add(sid)
        class_buckets[best_class].append(student.student_id)
        student.class_id = best_class
        classroom_assignments[student.student_id] = best_class

    # Save assignments to database
    db.session.commit()

    # === 8. Export to Neo4j ===
    # Clear existing data
    neo4j_service.run_query("MATCH (n) DETACH DELETE n")

    # Calculate classroom statistics
    classroom_stats = defaultdict(lambda: {"achievement": [], "wellbeing": []})
    for i, s in enumerate(students):
        classroom_stats[s.class_id]["achievement"].append(float(node_pred[i]))
        classroom_stats[s.class_id]["wellbeing"].append(float(s.psychological_distress))

    # Create student nodes in Neo4j
    for i, s in enumerate(students):
        class_id = s.class_id
        avg_gpa = np.mean(classroom_stats[class_id]["achievement"])
        avg_wellbeing = np.mean(classroom_stats[class_id]["wellbeing"])
        
        neo4j_service.run_query("""
        MERGE (s:Student {student_id: $id})
        SET s.achievement = $a,
            s.wellbeing = $w,
            s.classroom = $c,
            s.gender = $g,
            s.avg_class_gpa = $avg_gpa,
            s.avg_class_wellbeing = $avg_wellbeing
        """, {
            "id": s.student_id,
            "a": float(node_pred[i]),
            "w": float(s.psychological_distress),
            "c": class_id,
            "g": "Male" if s.encoded_gender == 1 else "Female",
            "avg_gpa": avg_gpa,
            "avg_class_wellbeing": avg_wellbeing
        })

    # Define relationship type labels
    EDGE_TYPE_LABELS = {0: "Friend", 1: "Neutral", 2: "Conflict"}

    # Create relationship edges in Neo4j
    for i in range(edge_index.size(1)):
        src = students[edge_index[0, i].item()].student_id
        tgt = students[edge_index[1, i].item()].student_id
        label = int(torch.argmax(edge_logits[i]).item())
        attn = float(attention[i])
        label_name = EDGE_TYPE_LABELS[label]

        neo4j_service.run_query("""
        MATCH (a:Student {student_id: $src})
        MATCH (b:Student {student_id: $tgt})
        MERGE (a)-[r:RELATIONSHIP]->(b)
        SET r.type = $type,
            r.label = $label,
            r.attention = $attention
        """, {
            "src": src,
            "tgt": tgt,
            "type": label,
            "label": label_name,
            "attention": attn
        })

    # === 9. Export D3-compatible JSON ===
    # Prepare node data for visualization
    d3_nodes = [{
        "id": s.student_id,
        "group": s.class_id,
        "class_id": s.class_id,
        "gender": "Male" if s.encoded_gender == 1 else "Female",
        "achievement": float(node_pred[i]),
        "wellbeing": float(s.psychological_distress),
        "avg_class_gpa": float(np.mean(classroom_stats[s.class_id]["achievement"])),
        "avg_class_wellbeing": float(np.mean(classroom_stats[s.class_id]["wellbeing"])),
        "embedding": x_emb[i].tolist()
    } for i, s in enumerate(students)]

    # Prepare edge data for visualization
    d3_links = [{
        "source": students[edge_index[0, i].item()].student_id,
        "target": students[edge_index[1, i].item()].student_id,
        "weight": float(attention[i]) if i < len(attention) else 0.0,
        "type": int(torch.argmax(edge_logits[i]).item()),
        "label": EDGE_TYPE_LABELS[int(torch.argmax(edge_logits[i]).item())]
    } for i in range(edge_index.size(1))]

    # Write visualization data to JSON file
    d3_path = "classroom_graph_d3.json"
    with open(d3_path, "w") as f:
        json.dump({"nodes": d3_nodes, "links": d3_links}, f, indent=2)

    # Prepare class summary statistics
    class_summary = [{
        "class_id": class_id,
        "avg_gpa": round(np.mean(stats["achievement"]), 2),
        "avg_wellbeing": round(np.mean(stats["wellbeing"]), 2),
        "student_count": len(stats["achievement"])
    } for class_id, stats in classroom_stats.items()]

    # Return summary of the allocation process
    return {
        "neo4j_status": "updated",
        "d3_path": d3_path,
        "num_students": len(students),
        "num_edges": edge_index.size(1),
        "class_summary": class_summary,
        "message": "✅ GAT allocation complete with edge type labels, Neo4j sync, D3 export, and class averages."
    }