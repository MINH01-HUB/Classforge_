# This module implements a Dual-Head Graph Attention Network (GAT) for simultaneous node and edge prediction.
# The model combines node-level regression (e.g., for academic performance prediction) with edge-level classification
# (e.g., for relationship type prediction between students).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.optim import Adam
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
import copy

class DualHeadGAT(nn.Module):
    """
    A dual-head Graph Attention Network that performs two tasks simultaneously:
    1. Node-level regression: Predicts continuous values for each node (e.g., academic performance)
    2. Edge-level classification: Predicts relationship types between nodes (e.g., friend/neutral/conflict)

    The model uses Graph Attention layers to learn node representations by attending to neighboring nodes,
    then applies separate prediction heads for node and edge tasks.

    Parameters:
    - in_channels (int): Number of input features per node
    - hidden_channels (int): Number of hidden units in GAT layers
    - out_node_dim (int): Output dimension for node prediction (usually 1 for regression)
    - out_edge_dim (int): Number of classes for edge classification
    - heads (int): Number of attention heads in each GAT layer
    - dropout (float): Dropout rate applied to GAT layers
    - num_classes (int): Number of relationship types to classify
    """
    def __init__(self, in_channels, hidden_channels, out_node_dim=1, out_edge_dim=3, heads=1, dropout=0.2, num_classes=3):
        super().__init__()
        self.dropout = dropout

        # First GAT layer: transforms input features to hidden representations
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        # Second GAT layer: further refines node representations
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True)

        # Node prediction head: predicts continuous values for each node
        self.node_predictor = nn.Linear(hidden_channels * heads, out_node_dim)
        # Edge prediction head: predicts relationship types between node pairs
        self.edge_predictor = nn.Bilinear(hidden_channels * heads, hidden_channels * heads, out_edge_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.

        Args:
        - x (Tensor): Node feature matrix [num_nodes, in_channels]
        - edge_index (LongTensor): Edge list in COO format [2, num_edges]

        Returns:
        - node_preds (Tensor): Node-level predictions [num_nodes]
        - edge_preds (Tensor): Edge-level logits [num_edges, out_edge_dim]
        - x (Tensor): Final node embeddings
        - attention_weights (List[float]): Averaged attention weights per edge
        """
        # First GAT layer with attention
        x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)  # Apply ELU activation
        # Second GAT layer with attention
        x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)  # Apply ELU activation

        # Generate node-level predictions
        node_preds = self.node_predictor(x).squeeze()
        # Generate edge-level predictions using node pairs
        row, col = edge_index
        edge_preds = self.edge_predictor(x[row], x[col])
        
        # Combine attention weights from both layers for interpretability
        attn_weights = (attn1[1].mean(dim=1) + attn2[1].mean(dim=1)) / 2
        return node_preds, edge_preds, x, attn_weights.tolist()

# later config with frontend to plot
# def plot_conf_matrix(cm, class_names=["Friend", "Neutral", "Conflict"]):
#     """Plot confusion matrix using seaborn heatmap."""
#     plt.figure(figsize=(5, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Edge Type Confusion Matrix")
#     plt.tight_layout()
#     plt.show()

def train_dual_head_gat(model, data, node_targets, edge_labels, epochs=100, lr=0.005, 
                        weight_decay=5e-4, val_split=0.2, test_split=0.2, early_stop_patience=10):
    """
    Train the DualHeadGAT model with both node regression and edge classification tasks.

    The training process:
    1. Splits data into train/validation/test sets for both nodes and edges
    2. Uses Adam optimizer with MSE loss for node regression and CrossEntropy loss for edge classification
    3. Implements early stopping based on validation F1 score
    4. Saves the best model weights and reports final test metrics

    Parameters:
    - model: DualHeadGAT instance
    - data: PyG Data object containing graph structure and features
    - node_targets: Tensor of target node scores (e.g., academic performance)
    - edge_labels: Tensor of edge labels (0 = Friend, 1 = Neutral, 2 = Conflict)
    - epochs: Number of training epochs
    - lr: Learning rate
    - weight_decay: Weight decay (L2 regularization)
    - val_split: Proportion of data used for validation
    - test_split: Proportion of data used for test
    - early_stop_patience: Early stopping after no improvement for these epochs
    """
    # Split node indices into train/validation/test sets
    num_nodes = data.x.size(0)
    node_indices = torch.arange(num_nodes)
    train_node_idx, temp_node_idx = train_test_split(node_indices, test_size=val_split+test_split, random_state=42)
    val_node_idx, test_node_idx = train_test_split(temp_node_idx, test_size=test_split/(val_split+test_split), random_state=42)

    # Split edge indices into train/validation/test sets
    num_edges = edge_labels.size(0)
    edge_indices = torch.arange(num_edges)
    edge_train_idx, temp_edge_idx = train_test_split(edge_indices, test_size=val_split+test_split, stratify=edge_labels.cpu(), random_state=42)
    edge_val_idx, edge_test_idx = train_test_split(temp_edge_idx, test_size=test_split/(val_split+test_split), stratify=edge_labels[temp_edge_idx].cpu(), random_state=42)

    # Create edge index tensors for each split
    train_edge_index = data.edge_index[:, edge_train_idx]
    val_edge_index = data.edge_index[:, edge_val_idx]
    test_edge_index = data.edge_index[:, edge_test_idx]

    # Initialize optimizer and loss functions
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()  # For node regression
    ce_loss = nn.CrossEntropyLoss()  # For edge classification

    # Training loop with early stopping
    best_model = None
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()

        # Forward pass
        node_pred, edge_logits, x_emb, _ = model(data.x, train_edge_index)

        # Calculate losses for both tasks
        loss_node = mse_loss(node_pred[train_node_idx], node_targets[train_node_idx])
        loss_edge = ce_loss(edge_logits, edge_labels[edge_train_idx])
        total_loss = loss_node + loss_edge
        total_loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            # Evaluate node predictions
            val_node_pred = node_pred[val_node_idx]
            val_rmse = mean_squared_error(node_targets[val_node_idx].cpu(), val_node_pred.cpu())

            # Evaluate edge predictions
            row, col = val_edge_index
            val_edge_logits = model.edge_predictor(x_emb[row], x_emb[col])
            val_edge_preds = val_edge_logits.argmax(dim=1).cpu()
            val_edge_true = edge_labels[edge_val_idx].cpu()

            # Calculate validation metrics
            val_acc = accuracy_score(val_edge_true, val_edge_preds)
            val_f1 = f1_score(val_edge_true, val_edge_preds, average='weighted', zero_division=0)
            val_cm = confusion_matrix(val_edge_true, val_edge_preds)

        # Print training progress
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss.item():.4f} | Val RMSE: {val_rmse:.3f} | Edge Acc: {val_acc:.3f} | F1: {val_f1:.3f}")
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_model = copy.deepcopy(model.state_dict())
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # Save best model
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "best_dual_head_gat.pth")

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        # Test node predictions
        test_node_pred, _, x_emb_test, _ = model(data.x, train_edge_index)
        test_rmse = mean_squared_error(node_targets[test_node_idx].cpu(), test_node_pred[test_node_idx].cpu())

        # Test edge predictions
        row, col = test_edge_index
        test_edge_logits = model.edge_predictor(x_emb_test[row], x_emb_test[col])
        test_edge_preds = test_edge_logits.argmax(dim=1).cpu()
        test_edge_true = edge_labels[edge_test_idx].cpu()

        # Calculate final test metrics
        test_acc = accuracy_score(test_edge_true, test_edge_preds)
        test_f1 = f1_score(test_edge_true, test_edge_preds, average='weighted', zero_division=0)
        test_cm = confusion_matrix(test_edge_true, test_edge_preds)

    # Print final results
    print("\n🧪 Final Test Results:")
    print(f"Node RMSE: {test_rmse:.3f} | Edge Acc: {test_acc:.3f} | F1: {test_f1:.3f}")
    print("✅ Training complete with proper data splits")
