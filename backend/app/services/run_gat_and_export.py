# This module implements the production version of the Graph Attention Network (GAT) execution pipeline.
# It handles student data processing, model training, classroom allocation, and data export to Neo4j and D3.js.
# The implementation includes persistent class assignments and conflict/friend-aware allocation strategies.

"""
🧼 Docker Command to Run a Clean Neo4j Instance:

Use this command in your terminal to start Neo4j cleanly:

    docker run -d \
      --name neo4j-test \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/testpassword \
      neo4j:5.15.0

This avoids permission issues on Windows and sets the password correctly.
Make sure your Flask app connects using:
    bolt://localhost:7687
    user: neo4j
    password: testpassword
"""

import torch
from torch_geometric.data import Data
from app.models.model import TransformedStudentData
from app.services.dual_head_gat import DualHeadGAT, train_dual_head_gat
from app.services.neo4j_service import neo4j_service
from app import db
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error

def run_gat_and_export(sliders):
    """
    Main function to run the GAT model and export results for classroom allocation.
    
    This function:
    1. Loads and processes student data
    2. Trains a DualHeadGAT model
    3. Generates classroom assignments considering academic performance, wellbeing, friendships, and conflicts
    4. Exports results to Neo4j and D3.js visualization format
    
    Args:
        sliders (dict): Dictionary containing weighting factors for different aspects:
            - academicBalance: Weight for academic performance (0-100)
            - wellbeingDistribution: Weight for wellbeing considerations (0-100)
            - friendshipRetention: Weight for maintaining friendships (0-100)
            - behavioralConsiderations: Weight for behavioral factors (0-100)
    
    Returns:
        dict: Summary of the execution including:
            - neo4j_status: Status of Neo4j database update
            - d3_path: Path to the generated D3 visualization file
            - num_students: Total number of students processed
            - num_edges: Number of relationships in the graph
            - message: Status message
    """
    # Load student data from database
    students = TransformedStudentData.query.all()
    if not students:
        raise ValueError("❌ No transformed student data found in the database.")

    # Reset existing class assignments to ensure clean allocation
    for s in students:
        s.class_id = None
    db.session.commit()

    # Prepare feature vectors and target values for the GAT model
    features, node_targets, edge_labels = [], [], []
    for s in students:
        features.append([
            s.encoded_gender,
            s.encoded_immigrant_status,
            s.ses,
            s.achievement,
            s.psychological_distress
        ])
        node_targets.append(s.achievement)

    # Convert features and targets to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float)
    node_targets = torch.tensor(node_targets, dtype=torch.float)

    # Create a simple chain graph structure for initial training
    # This will be refined by the GAT model during training
    src = list(range(len(students) - 1))
    tgt = list(range(1, len(students)))
    edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
    num_edges = edge_index.size(1)
    edge_labels = torch.randint(0, 3, (num_edges,))

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index)

    # Initialize and configure the DualHeadGAT model
    model = DualHeadGAT(
        in_channels=data.num_features,
        hidden_channels=16,
        out_node_dim=1,
        out_edge_dim=3,
        dropout=0.2,
        heads=1,
        num_classes=3
    )

    # Train the model
    train_dual_head_gat(model, data, node_targets, edge_labels)

    # Generate predictions using the trained model
    model.eval()
    with torch.no_grad():
        node_pred, edge_logits, x_emb, attention = model(data.x, data.edge_index)

    # Extract weights from slider inputs for scoring
    academic_w = sliders.get("academicBalance", 50) / 100.0
    wellbeing_w = sliders.get("wellbeingDistribution", 50) / 100.0
    friendship_w = sliders.get("friendshipRetention", 50) / 100.0
    behavior_w = sliders.get("behavioralConsiderations", 50) / 100.0

    # Calculate base scores considering academic performance and wellbeing
    distress_tensor = torch.tensor([s.psychological_distress for s in students])
    base_score = academic_w * node_pred - wellbeing_w * distress_tensor

    # Process GAT edge predictions to identify friendships and conflicts
    friend_score_map = torch.zeros(len(students))
    friendships = defaultdict(set)
    conflicts = defaultdict(set)

    # Analyze each edge in the graph
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        edge_type = torch.argmax(edge_logits[i]).item()  # 0 = Friend, 1 = Neutral, 2 = Conflict
        attn = attention[i]  # attention weight for this edge

        # Process friendship relationships
        if edge_type == 0:
            # Add attention-weighted influence to both nodes
            friend_score_map[src] += attn * friendship_w
            friend_score_map[tgt] += attn * friendship_w
            # Store mutual friendship
            friendships[src].add(tgt)
            friendships[tgt].add(src)
        # Process conflict relationships
        elif edge_type == 2:
            # Apply attention-weighted penalty to both nodes
            base_score[src] -= attn * behavior_w
            base_score[tgt] -= attn * behavior_w
            # Store mutual conflict
            conflicts[src].add(tgt)
            conflicts[tgt].add(src)

    # Combine scores for final ranking
    scores = base_score + friend_score_map
    sorted_indices = torch.argsort(scores, descending=True)

    # Initialize classroom allocation structures
    classroom_assignments = {}
    max_class_size = 40
    min_class_size = 35
    total_students = len(students)
    num_classes = max(1, total_students // min_class_size)
    class_buckets = {f"Class_{i+1}": [] for i in range(num_classes)}
    class_sets = {cid: set() for cid in class_buckets}

    # Assign students to classrooms
    for idx in sorted_indices:
        sid = idx.item()
        student = students[sid]

        # Find best class considering conflicts and friendships
        best_class = None
        best_friend_count = -1

        for cid, members in class_sets.items():
            # Skip full classes
            if len(members) >= max_class_size:
                continue
            # Skip classes with conflicts
            conflict = any(m in conflicts[sid] for m in members)
            if conflict:
                continue
            # Count friends in this class
            friend_count = sum(1 for m in members if m in friendships[sid])
            if friend_count > best_friend_count:
                best_class = cid
                best_friend_count = friend_count

        # If no suitable class found, assign to smallest class
        if best_class is None:
            best_class = min(class_sets.items(), key=lambda kv: len(kv[1]))[0]

        # Update assignments
        class_buckets[best_class].append(student.student_id)
        class_sets[best_class].add(sid)
        student.class_id = best_class
        classroom_assignments[student.student_id] = best_class

    # Clear existing data in Neo4j
    neo4j_service.run_query("MATCH (n) DETACH DELETE n")

    # Create student nodes in Neo4j
    for i, student in enumerate(students):
        neo4j_service.run_query(
            """
            MERGE (s:Student {student_id: $id})
            SET s.achievement = $a,
                s.wellbeing = $w,
                s.classroom = $c
            """,
            {"id": student.student_id, "a": float(node_pred[i]), "w": float(student.psychological_distress), "c": classroom_assignments[student.student_id]}
        )

    # Create relationship edges in Neo4j
    for i in range(edge_index.size(1)):
        src_idx, tgt_idx = edge_index[0, i].item(), edge_index[1, i].item()
        src_id = students[src_idx].student_id
        tgt_id = students[tgt_idx].student_id
        edge_type = int(torch.argmax(edge_logits[i]).item())
        attention_weight = float(attention[i]) if i < len(attention) else 0.0

        neo4j_service.run_query(
            f"""
            MATCH (a:Student {{student_id: $src_id}})
            MATCH (b:Student {{student_id: $tgt_id}})
            MERGE (a)-[r:RELATIONSHIP]->(b)
            SET r.type = $type, r.attention = $attention
            """,
            {"src_id": src_id, "tgt_id": tgt_id, "type": edge_type, "attention": attention_weight}
        )

    # Prepare node data for D3.js visualization
    d3_nodes = [{
        "id": s.student_id,
        "group": s.class_id,
        "achievement": float(node_pred[i]),
        "wellbeing": float(s.psychological_distress),
        "embedding": x_emb[i].tolist()
    } for i, s in enumerate(students)]

    # Prepare edge data for D3.js visualization
    d3_links = [{
        "source": students[edge_index[0, i].item()].student_id,
        "target": students[edge_index[1, i].item()].student_id,
        "weight": float(attention[i]) if i < len(attention) else 0.0,
        "type": int(torch.argmax(edge_logits[i]).item())
    } for i in range(edge_index.size(1))]

    # Export visualization data to JSON
    d3_data = {"nodes": d3_nodes, "links": d3_links}
    d3_path = "classroom_graph_d3_for_frontend.json"
    with open(d3_path, "w") as f:
        json.dump(d3_data, f, indent=2)

    # Return execution summary
    return {
        "neo4j_status": "updated",
        "d3_path": d3_path,
        "num_students": total_students,
        "num_edges": len(d3_links),
        "message": "GAT run complete with classroom assignment and database sync."
    }





# version 3.0.0
# # run_gat_and_export_prod.py (with persistent class_id and conflict/friend-aware allocation)

# import torch
# from torch_geometric.data import Data
# from app.models.model import TransformedStudentData
# from app.services.dual_head_gat import DualHeadGAT, train_dual_head_gat
# from app.services.neo4j_service import neo4j_service
# from app import db
# import json
# import numpy as np
# from collections import defaultdict


# def run_gat_and_export(sliders):
#     students = TransformedStudentData.query.all()
#     if not students:
#         raise ValueError("❌ No transformed student data found in the database.")

#     # Reset class assignments before reallocation
#     for s in students:
#         s.class_id = None
#     db.session.commit()

#     features, node_targets, edge_labels = [], [], []
#     for s in students:
#         features.append([
#             s.encoded_gender,
#             s.encoded_immigrant_status,
#             s.ses,
#             s.achievement,
#             s.psychological_distress
#         ])
#         node_targets.append(s.achievement)

#     x = torch.tensor(features, dtype=torch.float)
#     node_targets = torch.tensor(node_targets, dtype=torch.float)

#     src = list(range(len(students) - 1))
#     tgt = list(range(1, len(students)))
#     edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
#     num_edges = edge_index.size(1)
#     edge_labels = torch.randint(0, 3, (num_edges,))

#     data = Data(x=x, edge_index=edge_index)

#     model = DualHeadGAT(
#         in_channels=data.num_features,
#         hidden_channels=16,
#         out_node_dim=1,
#         out_edge_dim=3,
#         dropout=0.2,
#         heads=1,
#         num_classes=3
#     )

#     train_dual_head_gat(model, data, node_targets, edge_labels)

#     model.eval()
#     with torch.no_grad():
#         node_pred, edge_logits, x_emb, attention = model(data.x, data.edge_index)

#     academic_w = sliders.get("academicBalance", 50) / 100.0
#     wellbeing_w = sliders.get("wellbeingDistribution", 50) / 100.0
#     friendship_w = sliders.get("friendshipRetention", 50) / 100.0
#     behavior_w = sliders.get("behavioralConsiderations", 50) / 100.0

#     distress_tensor = torch.tensor([s.psychological_distress for s in students])
#     base_score = academic_w * node_pred - wellbeing_w * distress_tensor

#     friend_score_map = torch.zeros(len(students))
#     friendships = defaultdict(set)
#     conflicts = defaultdict(set)

#     for i in range(edge_index.size(1)):
#         src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
#         edge_type = torch.argmax(edge_logits[i]).item()
#         attn = attention[i]
#         if edge_type == 0:
#             friend_score_map[src] += attn * friendship_w
#             friend_score_map[tgt] += attn * friendship_w
#             friendships[src].add(tgt)
#             friendships[tgt].add(src)
#         elif edge_type == 2:
#             base_score[src] -= attn * behavior_w
#             base_score[tgt] -= attn * behavior_w
#             conflicts[src].add(tgt)
#             conflicts[tgt].add(src)

#     scores = base_score + friend_score_map
#     sorted_indices = torch.argsort(scores, descending=True)

#     classroom_assignments = {}
#     max_class_size = 40
#     min_class_size = 35
#     total_students = len(students)
#     num_classes = max(1, total_students // min_class_size)
#     class_buckets = {f"Class_{i+1}": [] for i in range(num_classes)}
#     class_sets = {cid: set() for cid in class_buckets}

#     for idx in sorted_indices:
#         sid = idx.item()
#         student = students[sid]

#         best_class = None
#         best_friend_count = -1

#         for cid, members in class_sets.items():
#             if len(members) >= max_class_size:
#                 continue
#             conflict = any(m in conflicts[sid] for m in members)
#             if conflict:
#                 continue
#             friend_count = sum(1 for m in members if m in friendships[sid])
#             if friend_count > best_friend_count:
#                 best_class = cid
#                 best_friend_count = friend_count

#         if best_class is None:
#             best_class = min(class_sets.items(), key=lambda kv: len(kv[1]))[0]

#         class_buckets[best_class].append(student.student_id)
#         class_sets[best_class].add(sid)
#         student.class_id = best_class
#         classroom_assignments[student.student_id] = best_class

#     # ✅ Already done earlier at the start of the function
# # (Removed duplicate reset block to avoid wiping after assignment)

#     neo4j_service.clear_graph()
#     for i, student in enumerate(students):
#         neo4j_service.create_student_node(student.student_id, {
#             "achievement": float(node_pred[i]),
#             "wellbeing": float(student.psychological_distress),
#             "classroom": classroom_assignments[student.student_id]
#         })

#     for i in range(edge_index.size(1)):
#         src_idx, tgt_idx = edge_index[0, i].item(), edge_index[1, i].item()
#         src_id = students[src_idx].student_id
#         tgt_id = students[tgt_idx].student_id
#         edge_type = int(torch.argmax(edge_logits[i]).item())
#         attention_weight = float(attention[i]) if i < len(attention) else 0.0
#         neo4j_service.create_edge(src_id, tgt_id, "RELATIONSHIP", {
#             "type": edge_type,
#             "attention": attention_weight
#         })

#     d3_nodes = [{
#         "id": s.student_id,
#         "group": s.class_id,
#         "achievement": float(node_pred[i]),
#         "wellbeing": float(s.psychological_distress),
#         "embedding": x_emb[i].tolist()
#     } for i, s in enumerate(students)]

#     d3_links = [{
#         "source": students[edge_index[0, i].item()].student_id,
#         "target": students[edge_index[1, i].item()].student_id,
#         "weight": float(attention[i]) if i < len(attention) else 0.0,
#         "type": int(torch.argmax(edge_logits[i]).item())
#     } for i in range(edge_index.size(1))]

#     d3_data = {"nodes": d3_nodes, "links": d3_links}
#     d3_path = "classroom_graph_d3.json"
#     with open(d3_path, "w") as f:
#         json.dump(d3_data, f, indent=2)

#     return {
#         "neo4j_status": "updated",
#         "d3_path": d3_path,
#         "num_students": total_students,
#         "num_edges": len(d3_links),
#         "message": "GAT run complete with classroom assignment and database sync."
#     }












#version 2
# # run_gat_and_export_prod.py

# import torch
# from torch_geometric.data import Data
# from app.models.model import TransformedStudentData
# from app.services.dual_head_gat import DualHeadGAT, train_dual_head_gat
# from app.services.neo4j_service import neo4j_service
# import json
# import os

# def run_gat_and_export(sliders):
#     """
#     Run GAT on preprocessed student data from the database and export results.

#     Args:
#         sliders (dict): Slider weights from frontend to guide logic (placeholder for future use)

#     Returns:
#         dict: Summary of result (Neo4j update status, D3 export path, counts)
#     """
#     # Step 1: Query transformed data from the DB
#     students = TransformedStudentData.query.all()
#     if not students:
#         raise ValueError("❌ No transformed student data found in the database.")

#     # Step 2: Build feature matrix and synthetic labels (targets + edge types placeholder)
#     features, node_targets, edge_labels = [], [], []
#     for s in students:
#         features.append([
#             s.encoded_gender,
#             s.encoded_immigrant_status,
#             s.ses,
#             s.achievement,
#             s.psychological_distress
#         ])
#         node_targets.append(s.achievement)

#     x = torch.tensor(features, dtype=torch.float)
#     node_targets = torch.tensor(node_targets, dtype=torch.float)

#     # Step 3: Create synthetic bidirectional edges and random edge labels
#     src = list(range(len(students) - 1))
#     tgt = list(range(1, len(students)))
#     edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
#     num_edges = edge_index.size(1)
#     edge_labels = torch.randint(0, 3, (num_edges,))  # placeholder: 0=Friend, 1=Neutral, 2=Conflict

#     # Step 4: Build PyG Data object
#     data = Data(x=x, edge_index=edge_index)

#     # Step 5: Initialize and train model
#     model = DualHeadGAT(
#         in_channels=data.num_features,
#         hidden_channels=16,
#         out_node_dim=1,
#         out_edge_dim=3,
#         dropout=0.2,
#         heads=1,
#         num_classes=3
#     )

#     train_dual_head_gat(model, data, node_targets, edge_labels)

#     # Step 6: Forward pass for final predictions
#     model.eval()
#     with torch.no_grad():
#         node_pred, edge_logits, x_emb, attention = model(data.x, data.edge_index)

#     # Step 7: Push results to Neo4j
#     neo4j_service.clear_graph()
#     for i, student in enumerate(students):
#         neo4j_service.create_student_node(student.student_id, {
#             "achievement": float(node_pred[i]),
#             "wellbeing": float(student.psychological_distress)
#         })

#     for i in range(edge_index.size(1)):
#         src_id = students[edge_index[0, i].item()].student_id
#         tgt_id = students[edge_index[1, i].item()].student_id
#         edge_type = int(torch.argmax(edge_logits[i]).item())
#         attention_weight = float(attention[i]) if i < len(attention) else 0.0

#         neo4j_service.create_edge(src_id, tgt_id, "RELATIONSHIP", {
#             "type": edge_type,
#             "attention": attention_weight
#         })

#     # Step 8: Export to D3-compatible JSON
#     d3_nodes = [{"id": s.student_id, "group": 1} for s in students]
#     d3_links = [{
#         "source": students[edge_index[0, i].item()].student_id,
#         "target": students[edge_index[1, i].item()].student_id,
#         "weight": float(attention[i]) if i < len(attention) else 0.0,
#         "type": int(torch.argmax(edge_logits[i]).item())
#     } for i in range(edge_index.size(1))]

#     d3_data = {"nodes": d3_nodes, "links": d3_links}
#     d3_path = "classroom_graph_d3_dual_head.json"
#     with open(d3_path, "w") as f:
#         json.dump(d3_data, f, indent=2)

#     return {
#         "neo4j_status": "updated",
#         "d3_path": d3_path,
#         "num_students": len(students),
#         "num_edges": len(d3_links)
#     }







#version 1.0.0
# import torch
# from torch_geometric.data import Data
# from app.models.model import TransformedStudentData
# from app.services.dual_head_gat import DualHeadGAT
# from app.services.neo4j_service import neo4j_service
# import json
# import os

# def run_gat_and_export(sliders):
#     """
#     Run GAT on preprocessed student data from the database and export results.
    
#     Args:
#         sliders (dict): Slider weights from frontend to guide logic (placeholder for future use)

#     Returns:
#         dict: Summary of result (Neo4j update status, D3 export path, counts)
#     """

#     # Step 1: Query transformed data from the DB
#     students = TransformedStudentData.query.all()
#     if not students:
#         raise ValueError("❌ No transformed student data found in the database.")

#     # Step 2: Build feature matrix
#     features = []
#     for s in students:
#         features.append([
#             s.encoded_gender,
#             s.encoded_immigrant_status,
#             s.ses,
#             s.achievement,
#             s.psychological_distress
#         ])
#     x = torch.tensor(features, dtype=torch.float)

#     # Step 3: Build synthetic edges (connect students sequentially, bidirectional)
#     src = list(range(len(students) - 1))
#     tgt = list(range(1, len(students)))
#     edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)

#     # Step 4: Prepare PyG Data object
#     data = Data(x=x, edge_index=edge_index)

#     # Step 5: Initialize DualHeadGAT
#     model = DualHeadGAT(
#         in_channels=data.num_features,
#         hidden_channels=16,
#         out_node_dim=1,
#         out_edge_dim=3,
#         dropout=0.2,# 0.2 standard,0.5 for overfitting
#         heads=1,#      
#         num_classes=3)
#     model.eval()

#     # Step 6: Forward pass
#     with torch.no_grad():
#         node_pred, edge_logits, attention = model(data.x, data.edge_index)

#     # Step 7: Push results to Neo4j
#     neo4j_service.clear_graph()

#     for i, student in enumerate(students):
#         neo4j_service.create_student_node(student.student_id, {
#             "achievement": float(node_pred[i][0]),
#             "wellbeing": float(student.psychological_distress)
#         })

#     for i in range(edge_index.size(1)):
#         src_id = students[edge_index[0, i].item()].student_id
#         tgt_id = students[edge_index[1, i].item()].student_id
#         edge_type = int(torch.argmax(edge_logits[i]).item())
#         attention_weight = float(attention[i].item()) if i < len(attention) else 0.0

#         neo4j_service.create_edge(src_id, tgt_id, "RELATIONSHIP", {
#             "type": edge_type,
#             "attention": attention_weight
#         })

#     # Step 8: Export to D3-compatible JSON
#     d3_nodes = [{"id": s.student_id, "group": 1} for s in students]
#     d3_links = [{
#         "source": students[edge_index[0, i].item()].student_id,
#         "target": students[edge_index[1, i].item()].student_id,
#         "weight": float(attention[i].item()) if i < len(attention) else 0.0,
#         "type": int(torch.argmax(edge_logits[i]).item())
#     } for i in range(edge_index.size(1))]

#     d3_data = {"nodes": d3_nodes, "links": d3_links}
#     d3_path = "classroom_graph_d3.json"
#     with open(d3_path, "w") as f:
#         json.dump(d3_data, f, indent=2)

#     return {
#         "neo4j_status": "updated",
#         "d3_path": d3_path,
#         "num_students": len(students),
#         "num_edges": len(d3_links)
#     }
