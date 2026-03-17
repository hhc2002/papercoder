# PaperCoder — arxiv:2502.09992

```python
# pip install torch transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

# Define a placeholder for the masked token
MASK_TOKEN = -1  # Assuming token IDs are non-negative integers

class MaskedPredictor(nn.Module):
    """
    A Transformer-based masked token predictor for LLaDA.
    This model is parameterized as p_theta(x_t | x_t) in the paper.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Initializes the MaskedPredictor.

        Args:
            vocab_size: The size of the vocabulary.
            d_model: The dimension of the model's embeddings.
            nhead: The number of attention heads in the Transformer encoder.
            num_layers: The number of Transformer encoder layers.
            dim_feedforward: The dimension of the feedforward network model in Transformer encoder layers.
            dropout: The dropout probability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(512, d_model) # Assuming a max sequence length of 512 for positional encoding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the masked predictor.

        Args:
            x_t: Input tensor representing partially masked sequences at time step t.
                 Shape: (batch_size, sequence_length)
            attention_mask: Optional attention mask to prevent attending to padding tokens.
                            Shape: (batch_size, sequence_length)

        Returns:
            Logits for predicting the original tokens at masked positions.
            Shape: (batch_size, sequence_length, vocab_size)
        """
        # TODO: Implement the forward pass of the masked predictor.
        # This involves:
        # 1. Embedding the input tokens (x_t).
        # 2. Adding positional encodings.
        # 3. Passing through the Transformer encoder.
        # 4. Projecting to vocabulary size to get logits.

        # Placeholder implementation:
        seq_len = x_t.size(1)
        positions = torch.arange(seq_len, device=x_t.device).unsqueeze(0)
        # Ensure positions are within the bounds of positional_encoding
        positions = positions % self.positional_encoding.num_embeddings

        token_emb = self.token_embedding(x_t)
        pos_emb = self.positional_encoding(positions)
        embedded = self.dropout(token_emb + pos_emb)

        # The Transformer expects a specific attention_mask format.
        # If a simple mask is provided (e.g., indicating padding), it needs to be converted.
        # For masked language modeling, we typically don't mask tokens themselves,
        # but rather use the attention mask to ignore padding.
        # If attention_mask is None, create a mask that attends to all tokens.
        if attention_mask is None:
            attention_mask = torch.ones(x_t.size(0), seq_len, device=x_t.device)
        # Transformer expects a mask where True means "ignore"
        transformer_attention_mask = (attention_mask == 0) # Assuming 0 means padding

        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=transformer_attention_mask)
        logits = self.fc_out(transformer_output)

        return logits

class LLaDA(nn.Module):
    """
    LLaDA: Large Language Diffusion Models.
    This class implements the LLaDA model, which defines a model distribution p_theta(x_0)
    through a forward and a reverse process.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Initializes the LLaDA model.

        Args:
            vocab_size: The size of the vocabulary.
            d_model: The dimension of the model's embeddings.
            nhead: The number of attention heads in the Transformer encoder.
            num_layers: The number of Transformer encoder layers.
            dim_feedforward: The dimension of the feedforward network model in Transformer encoder layers.
            dropout: The dropout probability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.masked_predictor = MaskedPredictor(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(self, x_t: torch.Tensor, t: float) -> torch.Tensor:
        """
        Forward pass of the LLaDA model, representing the reverse process.
        Predicts the original tokens x_0 given the partially masked sequence x_t at time t.

        Args:
            x_t: Input tensor representing partially masked sequences at time step t.
                 Shape: (batch_size, sequence_length)
            t: The current time step (a float between 0 and 1).

        Returns:
            Logits for predicting the original tokens at masked positions.
            Shape: (batch_size, sequence_length, vocab_size)
        """
        # The paper states: "The model uses cross-entropy loss, computed only on the masked tokens."
        # The masked_predictor is parameterized as p_theta(x_t | x_t).
        # This function essentially calls the masked_predictor.
        # The time step 't' might be used to condition the model in more advanced implementations,
        # but based on the provided description, it's directly passed to the predictor.
        # TODO: Consider if 't' needs to be explicitly encoded or used to condition the predictor.
        # For now, we assume the predictor implicitly handles the time step information or it's not explicitly conditioned.
        return self.masked_predictor(x_t)

    def loss_fn(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the LLaDA loss function.
        L(theta) = -E_{t, x_0, x_t} [ (1/t) * sum_{i=1}^{L} 1[x_t^i = M] * log p_theta(x_0^i | x_t^i) ]

        Args:
            x_0: The original clean sequences. Shape: (batch_size, sequence_length)
            x_t: The partially masked sequences at time step t. Shape: (batch_size, sequence_length)
            t: The time step for each sequence in the batch. Shape: (batch_size,)

        Returns:
            The calculated loss.
        """
        # TODO: Implement the LLaDA loss function.
        # This involves:
        # 1. Identifying the masked tokens in x_t.
        # 2. Getting predictions from the masked_predictor.
        # 3. Computing the cross-entropy loss only on the masked tokens.
        # 4. Applying the (1/t) weighting.

        logits = self.masked_predictor(x_t) # Shape: (batch_size, sequence_length, vocab_size)

        # Create a mask for the masked tokens.
        # Assuming MASK_TOKEN is used to represent masked positions in x_t.
        masked_token_mask = (x_t == MASK_TOKEN) # Shape: (batch_size, sequence_length)

        # Calculate cross-entropy loss.
        # We need to gather the predicted logits for the actual tokens in x_0 at the masked positions.
        # F.cross_entropy expects (N, C) and (N) or (N, C, d1, d2, ...) and (N, d1, d2, ...)
        # Here, N = batch_size * sequence_length, C = vocab_size.
        # We need to flatten the tensors and apply the mask.

        # Reshape for cross_entropy: (batch_size * sequence_length, vocab_size)
        logits_flat = logits.view(-1, self.vocab_size)
        # Reshape target: (batch_size * sequence_length)
        x_0_flat = x_0.view(-1)
        # Reshape mask: (batch_size * sequence_length)
        masked_token_mask_flat = masked_token_mask.view(-1)

        # Calculate loss only for masked tokens.
        # We can use F.cross_entropy with ignore_index or manually mask.
        # Let's manually mask for clarity with the (1/t) weighting.

        # Get the indices of the masked tokens
        masked_indices = torch.where(masked_token_mask_flat)[0]

        if masked_indices.numel() == 0:
            return torch.tensor(0.0, device=x_t.device, requires_grad=True) # No masked tokens to compute loss on

        # Gather the relevant logits and targets
        masked_logits = logits_flat[masked_indices]
        masked_x_0 = x_0_flat[masked_indices]

        # Compute cross-entropy loss for masked tokens
        ce_loss = F.cross_entropy(masked_logits, masked_x_0, reduction='none') # Shape: (num_masked_tokens,)

        # Apply the (1/t) weighting.
        # The time step 't' is a tensor of shape (batch_size,). We need to broadcast it correctly.
        # The loss is computed per token, so we need to associate each masked token with its batch's 't'.
        # This requires careful indexing or repeating 't'.

        # Let's re-think the loss calculation to align with the formula:
        # L = -E_{t, x_0, x_t} [ (1/t) * sum_{i=1}^{L} 1[x_t^i = M] * log p_theta(x_0^i | x_t^i) ]
        # This implies we should compute the sum for each sequence, then average over batches and time steps.

        # Alternative approach: Compute loss per sequence element, then mask and weight.
        # Log probability of the correct token at masked positions
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather the log probabilities for the actual tokens in x_0 at the masked positions
        gathered_log_probs = torch.gather(log_probs, -1, x_0.unsqueeze(-1)).squeeze(-1) # Shape: (batch_size, sequence_length)

        # Mask the log probabilities where tokens were not masked
        masked_log_probs = gathered_log_probs.masked_fill(~masked_token_mask, 0.0) # Set non-masked to 0

        # Apply the (1/t) weighting. t is (batch_size,)
        # We need to repeat t for each token in the sequence.
        t_weighted = t.unsqueeze(-1).repeat(1, x_0.size(1)) # Shape: (batch_size, sequence_length)
        # Avoid division by zero if t can be 0. The paper states t ~ U(0, 1].
        weighted_masked_log_probs = masked_log_probs / t_weighted

        # Sum over the sequence length
        sum_weighted_masked_log_probs = torch.sum(weighted_masked_log_probs, dim=-1) # Shape: (batch_size,)

        # The formula has a negative sign and an expectation.
        # We are calculating -sum(...) for each batch element, and then we'll average.
        loss = -sum_weighted_masked_log_probs

        # Average over the batch
        mean_loss = torch.mean(loss)

        return mean_loss

    def forward_process(self, x_0: torch.Tensor, t: float) -> torch.Tensor:
        """
        Simulates the forward diffusion process.
        Gradually masks tokens in x_0 until time step t.
        For t in (0, 1), each token is masked with probability t.

        Args:
            x_0: The original clean sequences. Shape: (batch_size, sequence_length)
            t: The target time step (a float between 0 and 1).

        Returns:
            The partially masked sequences x_t. Shape: (batch_size, sequence_length)
        """
        # TODO: Implement the forward process.
        # This involves creating x_t by masking tokens in x_0 with probability t.
        # Note: The paper's description of the forward process is a bit unusual:
        # "For t in (0, 1), the sequence x_t is partially masked, where each token is masked with probability t,
        # or remains unmasked with probability 1-t."
        # This is different from standard diffusion where noise is gradually added.
        # Here, it seems like a direct masking process.

        if t <= 0:
            return x_0 # At t=0, no masking

        batch_size, seq_len = x_0.shape
        # Create a mask tensor
        mask = torch.rand(batch_size, seq_len, device=x_t.device) < t
        # Apply the mask: replace tokens with MASK_TOKEN where mask is True
        x_t = torch.where(mask, torch.tensor(MASK_TOKEN, device=x_0.device, dtype=x_0.dtype), x_0)
        return x_t

    def reverse_process_sample(self, x_t: torch.Tensor, t: float, s: float) -> torch.Tensor:
        """
        Performs one step of the reverse diffusion process from time t to time s.
        Predicts masked tokens and re-masks a fraction of them.

        Args:
            x_t: Partially masked sequences at time step t. Shape: (batch_size, sequence_length)
            t: The current time step.
            s: The next time step (s < t).

        Returns:
            Partially masked sequences at time step s. Shape: (batch_size, sequence_length)
        """
        # TODO: Implement the reverse process step.
        # This involves:
        # 1. Feeding x_t into the masked_predictor to get predictions for masked tokens.
        # 2. Re-masking a fraction (s/t) of the predicted tokens.
        #    The paper mentions "low-confidence remasking strategy" for this.

        # Predict all masked tokens using the model
        logits = self.masked_predictor(x_t) # Shape: (batch_size, sequence_length, vocab_size)

        # Get the predicted token IDs (greedy sampling)
        predicted_tokens = torch.argmax(logits, dim=-1) # Shape: (batch_size, sequence_length)

        # Create a mask for tokens that were originally masked in x_t
        original_masked_mask = (x_t == MASK_TOKEN) # Shape: (batch_size, sequence_length)

        # Initialize x_s with x_t
        x_s = x_t.clone()

        # Update tokens that were masked in x_t with their predictions
        x_s = torch.where(original_masked_mask, predicted_tokens, x_s)

        # Apply remasking strategy: remask s/t of the predicted tokens.
        # The paper refers to "low-confidence remasking strategy" [23].
        # This means we need to identify the predicted tokens with the lowest confidence.
        # Confidence can be derived from the softmax probabilities of the predicted tokens.

        if s < t: # Only remask if moving to an earlier timestep
            # Calculate the number of tokens to remask.
            # The paper states: "s/t of predicted tokens with the lowest confidence are remarked"
            # Let's assume we need to remask approximately (s/t) * num_predicted_tokens.
            # A simpler interpretation might be to remask (s/t) fraction of *all* tokens that were predicted.

            # Get probabilities for the predicted tokens
            probs = F.softmax(logits, dim=-1)
            predicted_token_probs, _ = torch.max(probs, dim=-1) # Confidence of the predicted token

            # Mask the confidence scores where tokens were not originally masked
            masked_confidence = predicted_token_probs.masked_fill(~original_masked_mask, float('inf')) # Set non-masked to inf so they are not selected

            # Determine the number of tokens to remask
            num_predicted_tokens = torch.sum(original_masked_mask, dim=-1) # Number of tokens predicted per batch item
            num_to_remask = torch.floor(num_predicted_tokens * (s / t)).long() # Number to remask per batch item

            # Find the indices of tokens with the lowest confidence among the predicted ones
            # This is a bit tricky to do efficiently per batch item.
            # We can flatten, sort, and then select.

            # For simplicity, let's implement a random remasking first, then low-confidence.
            # Random remasking: remask s/t fraction of predicted tokens randomly.
            # TODO: Implement the actual low-confidence remasking strategy as described in Algorithm 5.
            # For now, a simplified random remasking:
            # Create a mask for remasking.
            # We need to select num_to_remask tokens from the originally masked tokens.

            # Simplified random remasking:
            # Create a mask for all tokens that were predicted.
            potential_remask_mask = original_masked_mask.clone().float()
            # Randomly set some of these to 0 (not to be remasked) to achieve the s/t ratio.
            # This is a rough approximation. A more precise method would involve sampling.

            # Let's use Algorithm 4's logic for random remasking as a placeholder.
            # Algorithm 4: "with probability s/t, r_0^i is set to M"
            # This implies that for each predicted token, we decide whether to remask it.
            remask_prob = s / t
            remask_decision = torch.rand(x_s.shape, device=x_s.device) < remask_prob

            # Apply remasking only to tokens that were predicted and where the decision is True.
            final_remask_mask = original_masked_mask & remask_decision
            x_s = torch.where(final_remask_mask, torch.tensor(MASK_TOKEN, device=x_s.device, dtype=x_s.dtype), x_s)

        return x_s

    def sample(self, prompt_tokens: torch.Tensor, num_steps: int = 1000) -> torch.Tensor:
        """
        Generates a sequence using the reverse diffusion process.

        Args:
            prompt_tokens: The initial prompt tokens (can be partially masked or a starting sequence).
                           Shape: (batch_size, sequence_length)
            num_steps: The number of reverse steps to take.

        Returns:
            The generated sequence. Shape: (batch_size, sequence_length)
        """
        # TODO: Implement the sampling procedure.
        # This involves starting from a fully masked sequence (or a sequence derived from the prompt)
        # and iteratively applying the reverse_process_sample function.
        # The paper mentions "iteratively predicting masked tokens to recover the data distribution,
        # moving from t=1 to t=0."

        batch_size, seq_len = prompt_tokens.shape
        # Initialize x_1 as a fully masked sequence.
        # The prompt might be used to condition the generation, or it could be the starting point.
        # Based on the description, it seems we start from a fully masked sequence and generate.
        # If prompt_tokens is meant to be a prefix, it needs to be handled.
        # For now, let's assume we start with a fully masked sequence of the same length as the prompt.
        x_t = torch.full((batch_size, seq_len), MASK_TOKEN, dtype=prompt_tokens.dtype, device=prompt_tokens.device)

        # Define the time steps for the reverse process.
        # We need to go from t=1 down to t=0.
        # The paper uses continuous time t in [0, 1]. For discrete steps, we can discretize this.
        # Let's assume num_steps corresponds to the number of intervals.
        # Time steps will be 1.0, 1.0 - 1/num_steps, ..., 1/num_steps.
        time_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=prompt_tokens.device)

        for i in range(num_steps):
            t = time_steps[i]
            s = time_steps[i+1]

            # If the prompt is meant to condition the generation, it should be fed into the model.
            # The current `masked_predictor` doesn't explicitly take a prompt.
            # TODO: Adapt the sampling to incorporate conditioning if needed.
            # For now, we generate unconditionally from a masked state.

            x_t = self.reverse_process_sample(x_t, t, s)

        # The final output x_0 is obtained after the last step.
        # Ensure no MASK_TOKEN remains if the process is complete.
        # If MASK_TOKEN remains, it implies the model failed to predict it.
        # We might need a final prediction pass or handle these cases.
        # For now, assume x_t at the end is the generated sequence.
        return x_t

    def evaluate_conditional_likelihood(self, p_0: torch.Tensor, r_0: torch.Tensor, n_mc: int = 10) -> torch.Tensor:
        """
        Evaluates the conditional log-likelihood using the equivalent form.
        -E_{L, r_0, r_t} [ (L/l) * sum_{i=1}^{l} I[r_t^i = M] * log p_0(r_t^i | p_0, r_t) ]

        Args:
            p_0: The prompt sequence. Shape: (batch_size, sequence_length)
            r_0: The response sequence. Shape: (batch_size, sequence_length)
            n_mc: The number of Monte Carlo estimations.

        Returns:
            The average conditional log-likelihood.
        """
        # TODO: Implement conditional log-likelihood evaluation.
        # This involves:
        # 1. Sampling a time step 'l' (sequence length for masking).
        # 2. Creating r_t by masking 'l' tokens from r_0.
        # 3. Using the masked predictor to estimate log p_0(r_t^i | p_0, r_t).
        # 4. Computing the weighted sum and averaging over MC samples.

        batch_size, seq_len = r_0.shape
        total_log_likelihood = 0.0

        for _ in range(n_mc):
            # Sample l uniformly from {1, 2, ..., L}
            l = torch.randint(1, seq_len + 1, (1,)).item()

            # Obtain r_t by uniformly sampling l tokens from r_0 without replacement for masking.
            # Create a mask for the l tokens to be masked.
            mask_indices = torch.randperm(seq_len)[:l]
            r_t = r_0.clone()
            r_t[:, mask_indices] = MASK_TOKEN

            # Predict the masked tokens using the model.
            # The model needs to be conditioned on the prompt p_0.
            # Assuming masked_predictor can handle conditioning implicitly or needs modification.
            # For now, let's assume the `forward` method of LLaDA can take p_0.
            # If not, the `masked_predictor` needs to be adapted.
            # The formula uses p_0(r_t^i | p_0, r_t), suggesting the model predicts based on prompt and current state.
            # Let's assume `self.masked_predictor` can take `p_0` as an argument or it's part of `r_t`'s context.
            # For now, we'll pass `r_t` and assume `p_0` is handled if needed by the predictor.
            # A common way is to concatenate p_0 and r_t, or use cross-attention.
            # Given the current `MaskedPredictor` signature, it only takes `x_t`.
            # TODO: Modify `MaskedPredictor` or `LLaDA.forward` to handle prompt conditioning.
            # For now, we'll use the existing predictor on `r_t`.

            # Get logits for predicting original tokens at masked positions.
            # We need to ensure the predictor is aware of the prompt `p_0`.
            # Let's assume a hypothetical `predict_masked` method that takes prompt.
            # If not, we'd need to modify `MaskedPredictor` to accept `p_0`.
            # For this implementation, we'll call the existing predictor on `r_t`.
            # This might not be fully accurate if prompt conditioning is crucial here.
            logits = self.masked_predictor(r_t) # Shape: (batch_size, sequence_length, vocab_size)

            # Calculate log probabilities for the original tokens in r_0 at the masked positions.
            log_probs = F.log_softmax(logits, dim=-1)
            gathered_log_probs = torch.gather(log_probs, -1, r_0.unsqueeze(-1)).squeeze(-1) # Shape: (batch_size, sequence_length)

            # Create a mask for the originally masked tokens in r_t.
            masked_token_mask = (r_t == MASK_TOKEN) # Shape: (batch_size, sequence_length)

            # Mask the log probabilities where tokens were not masked.
            masked_log_probs = gathered_log_probs.masked_fill(~masked_token_mask, 0.0)

            # Calculate the sum: (L/l) * sum_{i=1}^{l} I[r_t^i = M] * log p_0(...)
            # Here, L is seq_len, l is the sampled number of masked tokens.
            # The sum is over the sequence length.
            sum_term = torch.sum(masked_log_probs, dim=-1) # Sum over sequence length
            weighted_sum = (seq_len / l) * sum_term # Shape: (batch_size,)

            # The formula is negative expectation. We are calculating the term inside the expectation.
            # We will average these terms over MC samples and then take the negative.
            total_log_likelihood += weighted_sum.mean().item() # Average over batch and add to total

        # Average over Monte Carlo estimations and take the negative.
        avg_log_likelihood = - (total_log_likelihood / n_mc)
        return avg_log_likelihood

# --- Helper functions for training and data preparation ---

def create_masked_data(x_0: torch.Tensor, t_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates masked data for training.
    Args:
        x_0: Original clean sequences. Shape: (batch_size, sequence_length)
        t_values: Time steps for each sequence. Shape: (batch_size,)
    Returns:
        x_t: Partially masked sequences. Shape: (batch_size, sequence_length)
        t_values: Time steps. Shape: (batch_size,)
        mask: Mask indicating which tokens were masked. Shape: (batch_size, sequence_length)
    """
    # TODO: Implement the masking strategy for creating x_t based on t_values.
    # This should align with the forward_process logic.
    batch_size, seq_len = x_0.shape
    mask = torch.rand(batch_size, seq_len, device=x_0.device) < t_values.unsqueeze(-1)
    x_t = torch.where(mask, torch.tensor(MASK_TOKEN, device=x_0.device, dtype=x_0.dtype), x_0)
    return x_t, t_values, mask

def train_step(model: LLaDA, optimizer: torch.optim.Optimizer, batch: Dict[str, torch.Tensor], device: torch.device) -> float:
    """
    Performs a single training step.

    Args:
        model: The LLaDA model.
        optimizer: The optimizer.
        batch: A dictionary containing the training data ('x_0', 't').
        device: The device to perform computations on.

    Returns:
        The loss for this training step.
    """
    model.train()
    x_0 = batch['x_0'].to(device)
    t = batch['t'].to(device) # Time steps for each sequence in the batch

    # Create masked data x_t
    x_t, _, _ = create_masked_data(x_0, t)

    # Calculate loss
    loss = model.loss_fn(x_0, x_t, t)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate_step(model: LLaDA, batch: Dict[str, torch.Tensor], device: torch.device) -> float:
    """
    Performs a single evaluation step.

    Args:
        model: The LLaDA model.
        batch: A dictionary containing the evaluation data ('x_0', 't').
        device: The device to perform computations on.

    Returns:
        The loss for this evaluation step.
    """
    model.eval()
    x_0 = batch['x_0'].to(device)
    t = batch['t'].to(device)

    # Create masked data x_t
    x_t, _, _ = create_masked_data(x_0, t)

    with torch.no_grad():
        loss = model.loss_fn(x_0, x_t, t)

    return loss.item()

def generate_sample(model: LLaDA, prompt_tokens: torch.Tensor, num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Generates a sample using the LLaDA model.

    Args:
        model: The LLaDA model.
        prompt_tokens: The initial prompt tokens. Shape: (batch_size, sequence_length)
        num_steps: The number of reverse diffusion steps.
        device: The device to perform computations on.

    Returns:
        The generated sequence tokens. Shape: (batch_size, sequence_length)
    """
    model.eval()
    prompt_tokens = prompt_tokens.to(device)
    generated_tokens = model.sample(prompt_tokens, num_steps=num_steps)
    return generated_tokens

def evaluate_likelihood(model: LLaDA, prompt: torch.Tensor, response: torch.Tensor, n_mc: int, device: torch.device) -> float:
    """
    Evaluates the conditional log-likelihood of a response given a prompt.

    Args:
        model: The LLaDA model.
        prompt: The prompt tokens. Shape: (batch_size, sequence_length)
        response: The response tokens. Shape: (batch_size, sequence_length)
        n_mc: Number of Monte Carlo samples.
        device: The device to perform computations on.

    Returns:
        The average conditional log-likelihood.
    """
    model.eval()
    prompt = prompt.to(device)
    response = response.to(device)
    likelihood = model.evaluate_conditional_likelihood(prompt, response, n_mc=n_mc)
    return likelihood

# --- Example Usage ---
if __name__ == "__main__":
    # Configuration
    VOCAB_SIZE = 10000
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    BATCH_SIZE = 4
    SEQ_LEN = 64
    NUM_REVERSE_STEPS = 100 # For sampling
    NUM_MC_ESTIMATIONS = 5 # For likelihood evaluation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model = LLaDA(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- Dummy Data for Demonstration ---
    # In a real scenario, you would load your dataset here.
    dummy_x_0 = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    dummy_t = torch.rand(BATCH_SIZE) # Random time steps for the batch

    dummy_batch = {'x_0': dummy_x_0, 't': dummy_t}

    # --- Training Step Example ---
    print("Running a dummy training step...")
    train_loss = train_step(model, optimizer, dummy_batch, device)
    print(f"Dummy Training Loss: {train_loss:.4f}")

    # --- Evaluation Step Example ---
    print("Running a dummy evaluation step...")
    eval_loss = evaluate_step(model, dummy_batch, device)
    print(f"Dummy Evaluation Loss: {eval_loss:.4f}")

    # --- Sampling Example ---
    print("Running a dummy sampling process...")
    # Create a dummy prompt (can be partially masked or just a starting sequence)
    dummy_prompt = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN)) # Batch size 1 for sampling example
    generated_sequence = generate_sample(