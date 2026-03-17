# PaperCoder — Attention Is All You Need

# pip install torch numpy

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple

# Dummy implementations for demonstration purposes
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # TODO: Implement actual MultiHeadAttention layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass for MultiHeadAttention
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.linear1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_hidden_dim, embed_dim)
        # TODO: Implement actual PositionwiseFeedForward layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass for PositionwiseFeedForward
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # TODO: Add residual connections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass for TransformerEncoderLayer with residual connections and layer normalization
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class TransformerModel(nn.Module):
    def __init__(self, num_encoder_layers: int, embed_dim: int, num_heads: int, ff_hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, num_classes)
        # TODO: Add input embedding and positional encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass for the full Transformer model
        for layer in self.layers:
            x = layer(x)
        # Assuming x is the output of the last encoder layer, we need to aggregate it
        # For simplicity, let's take the first token's representation if it's sequence data
        # or average if it's a fixed-size representation.
        # This part is highly dependent on the specific task (e.g., classification, sequence generation)
        # For image classification, typically a CLS token is used or global average pooling.
        # Here, we'll assume a simplified scenario where we take the first element for demonstration.
        if x.dim() > 2: # If it's a sequence, take the first element's representation
             x = x[:, 0, :]
        elif x.dim() > 1: # If it's already a batch of vectors
             pass # Use as is
        else: # If it's a single vector
             x = x.unsqueeze(0) # Add batch dimension

        return self.fc_out(x)

def build_model(config: Dict[str, Any]) -> TransformerModel:
    """
    Builds a Transformer model based on the provided configuration.

    Args:
        config: A dictionary containing model configuration parameters.
                Expected keys: 'num_encoder_layers', 'embed_dim', 'num_heads',
                               'ff_hidden_dim', 'num_classes', 'dropout'.
                               'attention_layers' and 'feedforward_layers' are used
                               to control the presence of these components.

    Returns:
        A TransformerModel instance.
    """
    num_encoder_layers = config.get('num_encoder_layers', 6)
    embed_dim = config.get('embed_dim', 512)
    num_heads = config.get('num_heads', 8)
    ff_hidden_dim = config.get('ff_hidden_dim', 2048)
    num_classes = config.get('num_classes', 10)
    dropout = config.get('dropout', 0.1)

    # Adjust the number of layers based on the config
    # This is a simplified way to control layer presence.
    # A more robust implementation might involve conditional layer creation.
    actual_num_encoder_layers = 0
    if config.get('attention_layers', 0) > 0:
        actual_num_encoder_layers = num_encoder_layers # Use the configured number of layers if attention is enabled

    # If only feedforward layers are requested, we still need a TransformerEncoderLayer structure
    # but we might disable the attention part within it if possible, or just have it as a placeholder.
    # For this example, we'll assume TransformerEncoderLayer always has both, and we control
    # the *number* of such layers. If attention_layers is 0, we effectively have no Transformer layers.
    # If feedforward_layers is 0, we'd need to modify TransformerEncoderLayer to skip FF.

    # For this specific experiment, we are testing the *necessity* of FF layers.
    # Model A: Only FF layers. This is tricky with the current TransformerEncoderLayer.
    # A true "only FF" model might not use the TransformerEncoderLayer structure at all.
    # For this skeleton, we'll interpret "only FF" as having FF layers but no attention.
    # This requires modifying TransformerEncoderLayer or creating a different layer type.
    # Let's assume for now that if attention_layers is 0, we don't build TransformerEncoderLayers.
    # And if feedforward_layers is 0, we'd need a modified TransformerEncoderLayer.

    # Given the prompt's description, it implies comparing:
    # 1. Model with FF layers only (no attention)
    # 2. Model with Attention layers only (no FF)
    # 3. Model with both Attention and FF layers

    # This requires a flexible TransformerEncoderLayer or different layer types.
    # Let's adapt the TransformerModel to handle this based on the config flags.

    class FlexibleTransformerEncoderLayer(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, use_attention: bool = True, use_feedforward: bool = True):
            super().__init__()
            self.use_attention = use_attention
            self.use_feedforward = use_feedforward

            if self.use_attention:
                self.attention = MultiHeadAttention(embed_dim, num_heads)
                self.dropout1 = nn.Dropout(dropout)
                self.norm1 = nn.LayerNorm(embed_dim)
            else:
                self.attention = None
                self.dropout1 = None
                self.norm1 = None

            if self.use_feedforward:
                self.feed_forward = PositionwiseFeedForward(embed_dim, ff_hidden_dim)
                self.dropout2 = nn.Dropout(dropout)
                self.norm2 = nn.LayerNorm(embed_dim)
            else:
                self.feed_forward = None
                self.dropout2 = None
                self.norm2 = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            if self.use_attention:
                attn_output = self.attention(x)
                x = self.norm1(residual + self.dropout1(attn_output))
                residual = x # Update residual for the next potential layer

            if self.use_feedforward:
                ff_output = self.feed_forward(x)
                x = self.norm2(residual + self.dropout2(ff_output))
            
            return x

    # Determine the number of layers to actually build
    num_attention_layers = config.get('attention_layers', 0)
    num_feedforward_layers = config.get('feedforward_layers', 0)

    # The experiment setup implies we are comparing configurations of *encoder blocks*.
    # Let's assume 'num_encoder_layers' in the config refers to the *total* number of blocks.
    # And 'attention_layers' and 'feedforward_layers' control *which components are active within those blocks*.
    # However, the pseudocode suggests setting the *count* of layers to 0.
    # This implies we might build different *types* of models.

    # Let's follow the pseudocode's implication more directly:
    # Model A: Only FF layers. This means we need a model structure that *only* has FF.
    # Model B: Only Attention layers. This means we need a model structure that *only* has Attention.
    # Model C: Both. Standard Transformer.

    # This requires a more dynamic model builder.

    model_type = config.get('model_type', 'standard') # 'standard', 'attention_only', 'ff_only'

    if model_type == 'standard':
        # Standard Transformer with both attention and FF
        layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
    elif model_type == 'attention_only':
        # Model with only attention layers
        # This requires a modified TransformerEncoderLayer or a different structure
        # For simplicity, let's assume we can disable FF in the layer
        layers = nn.ModuleList([
            FlexibleTransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout, use_attention=True, use_feedforward=False)
            for _ in range(num_encoder_layers) # Use num_encoder_layers as the count of attention blocks
        ])
        if num_encoder_layers == 0: # If attention_layers was set to 0
             layers = nn.ModuleList() # Empty list
    elif model_type == 'ff_only':
        # Model with only feedforward layers
        # This requires a modified TransformerEncoderLayer or a different structure
        layers = nn.ModuleList([
            FlexibleTransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, dropout, use_attention=False, use_feedforward=True)
            for _ in range(num_encoder_layers) # Use num_encoder_layers as the count of FF blocks
        ])
        if num_encoder_layers == 0: # If feedforward_layers was set to 0
             layers = nn.ModuleList() # Empty list
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # The TransformerModel class needs to be adapted to handle these different layer types.
    # Let's redefine TransformerModel to accept a list of layers.

    class DynamicTransformerModel(nn.Module):
        def __init__(self, encoder_layers: nn.ModuleList, embed_dim: int, num_classes: int):
            super().__init__()
            self.layers = encoder_layers
            self.fc_out = nn.Linear(embed_dim, num_classes)
            # TODO: Add input embedding and positional encoding if needed for the task

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # TODO: Implement forward pass for the full model, including embeddings and positional encoding
            for layer in self.layers:
                x = layer(x)

            # Aggregation logic (same as before)
            if x.dim() > 2:
                 x = x[:, 0, :]
            elif x.dim() > 1:
                 pass
            else:
                 x = x.unsqueeze(0)

            return self.fc_out(x)

    # Rebuild based on the dynamic model
    model = DynamicTransformerModel(layers, embed_dim, num_classes)
    return model


def train(model: nn.Module, train_data: Any, epochs: int = 5, lr: float = 0.001) -> None:
    """
    Trains the given model.

    Args:
        model: The PyTorch model to train.
        train_data: The training dataset (assumed to be a DataLoader or similar iterable).
        epochs: Number of training epochs.
        lr: Learning rate.
    """
    print(f"Starting training for {epochs} epochs...")
    # TODO: Implement actual training loop
    # This includes defining loss function, optimizer, and iterating through data.
    criterion = nn.CrossEntropyLoss() # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Example optimizer

    model.train() # Set model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        # Assuming train_data is a DataLoader yielding (inputs, labels)
        for i, data in enumerate(train_data, 0):
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        print(f"Epoch {epoch+1} finished.")
    print("Finished Training")


def evaluate(model: nn.Module, test_data: Any) -> Dict[str, float]:
    """
    Evaluates the model on the test dataset.

    Args:
        model: The PyTorch model to evaluate.
        test_data: The test dataset (assumed to be a DataLoader or similar iterable).

    Returns:
        A dictionary containing performance metrics (e.g., accuracy).
    """
    print("Starting evaluation...")
    # TODO: Implement actual evaluation loop
    # This includes calculating metrics like accuracy, precision, recall, etc.
    correct = 0
    total = 0
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        # Assuming test_data is a DataLoader yielding (inputs, labels)
        for data in test_data:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
    return {"accuracy": accuracy}

def run_experiment(dataset: Any, model_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Runs the experiment to compare different model configurations.

    Args:
        dataset: The dataset containing training and testing data.
                 Expected to have 'train' and 'test' attributes, which are DataLoaders.
        model_config: Base configuration for the models.

    Returns:
        A dictionary containing performance metrics for each model configuration.
    """
    performance_metrics: Dict[str, Dict[str, float]] = {}

    # --- Configuration for Model A: Only Feedforward Layers ---
    print("\n--- Configuring Model A (Feedforward Only) ---")
    model_A_config = model_config.copy()
    # According to pseudocode, set attention_layers to 0 and feedforward_layers to > 0
    # We interpret this as building a model with FF components only.
    model_A_config['model_type'] = 'ff_only'
    # Ensure we have at least one layer if the base config didn't specify counts
    if 'num_encoder_layers' not in model_A_config or model_A_config['num_encoder_layers'] == 0:
         model_A_config['num_encoder_layers'] = model_config.get('ff_hidden_dim', 1) // 512 # Heuristic
    print(f"Model A config: {model_A_config}")

    # --- Configuration for Model B: Only Attention Layers ---
    print("\n--- Configuring Model B (Attention Only) ---")
    model_B_config = model_config.copy()
    # According to pseudocode, set feedforward_layers to 0 and attention_layers to > 0
    # We interpret this as building a model with Attention components only.
    model_B_config['model_type'] = 'attention_only'
    if 'num_encoder_layers' not in model_B_config or model_B_config['num_encoder_layers'] == 0:
         model_B_config['num_encoder_layers'] = model_config.get('num_heads', 1) # Heuristic
    print(f"Model B config: {model_B_config}")

    # --- Configuration for Model C: Standard Transformer (Both) ---
    print("\n--- Configuring Model C (Standard Transformer) ---")
    model_C_config = model_config.copy()
    # Standard configuration, both attention and FF are active.
    model_C_config['model_type'] = 'standard'
    if 'num_encoder_layers' not in model_C_config or model_C_config['num_encoder_layers'] == 0:
         model_C_config['num_encoder_layers'] = 6 # Default Transformer encoder layers
    print(f"Model C config: {model_C_config}")

    # --- Training and Evaluation ---

    # Model A
    print("\n--- Training and Evaluating Model A ---")
    model_A = build_model(model_A_config)
    train(model_A, dataset.train)
    performance_metrics['model_A'] = evaluate(model_A, dataset.test)

    # Model B
    print("\n--- Training and Evaluating Model B ---")
    model_B = build_model(model_B_config)
    train(model_B, dataset.train)
    performance_metrics['model_B'] = evaluate(model_B, dataset.test)

    # Model C
    print("\n--- Training and Evaluating Model C ---")
    model_C = build_model(model_C_config)
    train(model_C, dataset.train)
    performance_metrics['model_C'] = evaluate(model_C, dataset.test)

    # --- Analysis ---
    print("\n--- Experiment Results ---")
    print(f"Model A (FF Only) Performance: {performance_metrics.get('model_A', 'N/A')}")
    print(f"Model B (Attention Only) Performance: {performance_metrics.get('model_B', 'N/A')}")
    print(f"Model C (Standard Transformer) Performance: {performance_metrics.get('model_C', 'N/A')}")

    # TODO: Add detailed analysis and conclusion based on the comparison of metrics.
    # For example, if model_A performs poorly and model_B performs poorly,
    # but model_C performs well, it supports the hypothesis that both are needed.
    # If model_A performs well, it suggests FF layers alone can be sufficient for some tasks.
    # If model_B performs well, it suggests Attention layers alone can be sufficient.

    return performance_metrics

if __name__ == '__main__':
    # This section is for demonstration and testing purposes.
    # In a real scenario, you would load a proper dataset (e.g., CIFAR-10, ImageNet).

    print("Setting up dummy dataset and configuration...")

    # Dummy Dataset (replace with actual DataLoader)
    class DummyDataset:
        def __init__(self, num_samples=1000, input_dim=784, num_classes=10):
            self.train_data = []
            self.test_data = []

            # Generate dummy training data
            for _ in range(num_samples):
                # Simulate input features (e.g., flattened image) and a label
                features = torch.randn(input_dim)
                label = torch.randint(0, num_classes, (1,)).item()
                self.train_data.append((features, label))

            # Generate dummy test data
            for _ in range(num_samples // 5):
                features = torch.randn(input_dim)
                label = torch.randint(0, num_classes, (1,)).item()
                self.test_data.append((features, label))

            # Wrap in DataLoaders for the train/evaluate functions
            self.train = torch.utils.data.DataLoader(self.train_data, batch_size=32, shuffle=True)
            self.test = torch.utils.data.DataLoader(self.test_data, batch_size=32, shuffle=False)

    # Dummy Model Configuration
    base_model_config: Dict[str, Any] = {
        'embed_dim': 128,       # Embedding dimension
        'num_heads': 4,         # Number of attention heads
        'ff_hidden_dim': 256,   # Hidden dimension in feedforward network
        'num_classes': 10,      # Number of output classes
        'dropout': 0.1,         # Dropout rate
        # 'num_encoder_layers' will be set dynamically based on model_type in run_experiment
    }

    # Instantiate dummy dataset
    # Adjust input_dim to match embed_dim if necessary, or add an embedding layer
    # For simplicity, let's assume input_dim matches embed_dim for this dummy example
    dummy_dataset = DummyDataset(input_dim=base_model_config['embed_dim'])

    # Run the experiment
    # The pseudocode implies setting specific counts for attention/feedforward layers.
    # Let's adapt the run_experiment to reflect this more directly.

    # --- Experiment Setup based on Pseudocode ---
    # We need to define the number of layers for each component type.
    # The pseudocode suggests setting counts to 0 for specific models.

    # Model A: Only Feedforward Layers
    # Pseudocode: attention_layers = 0, feedforward_layers = num_feedforward_layers
    # Let's set num_feedforward_layers = 2 for this example.
    model_A_params = base_model_config.copy()
    model_A_params['model_type'] = 'ff_only'
    model_A_params['num_encoder_layers'] = 2 # Number of FF blocks
    print(f"\n--- Running Experiment for Model A (FF Only, {model_A_params['num_encoder_layers']} layers) ---")
    model_A = build_model(model_A_params)
    train(model_A, dummy_dataset.train, epochs=2) # Reduced epochs for faster demo
    perf_A = evaluate(model_A, dummy_dataset.test)

    # Model B: Only Attention Layers
    # Pseudocode: feedforward_layers = 0, attention_layers = num_attention_layers
    # Let's set num_attention_layers = 2 for this example.
    model_B_params = base_model_config.copy()
    model_B_params['model_type'] = 'attention_only'
    model_B_params['num_encoder_layers'] = 2 # Number of Attention blocks
    print(f"\n--- Running Experiment for Model B (Attention Only, {model_B_params['num_encoder_layers']} layers) ---")
    model_B = build_model(model_B_params)
    train(model_B, dummy_dataset.train, epochs=2) # Reduced epochs for faster demo
    perf_B = evaluate(model_B, dummy_dataset.test)

    # Model C: Standard Transformer (Both)
    # Pseudocode: attention_layers = num_attention_layers, feedforward_layers = num_feedforward_layers
    # Let's set num_attention_layers = 2, num_feedforward_layers = 2 for this example.
    # This corresponds to 2 standard TransformerEncoderLayers.
    model_C_params = base_model_config.copy()
    model_C_params['model_type'] = 'standard'
    model_C_params['num_encoder_layers'] = 2 # Number of standard Transformer blocks
    print(f"\n--- Running Experiment for Model C (Standard Transformer, {model_C_params['num_encoder_layers']} layers) ---")
    model_C = build_model(model_C_params)
    train(model_C, dummy_dataset.train, epochs=2) # Reduced epochs for faster demo
    perf_C = evaluate(model_C, dummy_dataset.test)

    # Collect results
    all_performance_metrics = {
        'model_A': perf_A,
        'model_B': perf_B,
        'model_C': perf_C,
    }

    print("\n--- Final Experiment Summary ---")
    print(f"Model A (FF Only) Performance: {all_performance_metrics['model_A']}")
    print(f"Model B (Attention Only) Performance: {all_performance_metrics['model_B']}")
    print(f"Model C (Standard Transformer) Performance: {all_performance_metrics['model_C']}")

    # Analysis based on the dummy results (likely random and not meaningful)
    # In a real experiment, you would analyze these metrics.
    print("\n--- Analysis ---")
    print("Comparing the performance metrics above would reveal the impact of feedforward layers.")
    print("If Model A and Model B perform significantly worse than Model C, it suggests")
    print("that both attention and feedforward layers are crucial for achieving high performance,")
    print("as demonstrated in the 'Attention Is All You Need' paper for sequence transduction tasks.")
    print("For image tasks, the interaction between self-attention and feedforward layers")
    print("within the Transformer architecture is key.")

```