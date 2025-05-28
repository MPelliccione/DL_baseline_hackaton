import torch
import torch.nn.functional as F

def GCOD_loss(output, target, model_embeddings, temperature=0.5, lambda_contrast=0.5):
    """
    Implementation of Graph Contrastive Outlier Detection loss
    
    Args:
        output: Model predictions (logits)
        target: Ground truth labels
        model_embeddings: Graph embeddings from the model
        temperature: Temperature parameter for contrastive learning (default: 0.5)
        lambda_contrast: Weight for contrastive loss component (default: 0.5)
    
    Returns:
        Combined loss value
    """
    # Classification loss (Cross Entropy)
    ce_loss = F.cross_entropy(output, target)
    
    # Contrastive loss
    batch_size = model_embeddings.size(0)
    
    # Normalize embeddings
    norm_embeddings = F.normalize(model_embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)
    
    # Create mask for positive pairs (same class)
    mask = target.expand(batch_size, batch_size).eq(target.expand(batch_size, batch_size).t())
    mask = mask.float()
    
    # Remove diagonal elements
    mask = mask.fill_diagonal_(0)
    
    # Scale similarities by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # Compute positive and negative pairs
    exp_sim = torch.exp(similarity_matrix)
    pos_sim = torch.sum(exp_sim * mask, dim=1)
    neg_sim = torch.sum(exp_sim * (1 - mask), dim=1)
    
    # Compute contrastive loss
    contrastive_loss = -torch.mean(torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)))
    
    # Combine losses
    total_loss = (1 - lambda_contrast) * ce_loss + lambda_contrast * contrastive_loss
    
    return total_loss