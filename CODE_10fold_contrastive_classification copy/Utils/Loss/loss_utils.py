import torch

### Contrastive Loss
# y=1 (성공 vs 실패) → 거리를 최대화
# y=0 (성공 vs 성공) → 거리를 최소화
def contrastive_loss(z1, z2, y, margin=1.0):
    distance = torch.nn.functional.pairwise_distance(z1, z2)
    loss = torch.mean((1 - y) * torch.pow(distance, 2) + y * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    return loss

### Triplet Loss
# Triplet Network는 (Anchor, Positive, Negative) 구조를 사용하여,
# Anchor & Positive (성공 vs 성공) → 가까워지도록
# Anchor & Negative (성공 vs 실패) → 멀어지도록 학습
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
    neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)
    loss = torch.mean(torch.clamp(pos_dist - neg_dist + margin, min=0.0))
    return loss


def mape_loss(preds, targets, epsilon=1e-8):
    """
    Mean Absolute Percentage Error (MAPE)
    
    Args:
        preds (Tensor): 모델의 예측값 (예: [20, 30, 50])
        targets (Tensor): 실제 값 (예: [22, 28, 55])
        epsilon (float): 0으로 나누는 문제 방지 (기본값 1e-8)
    
    Returns:
        Tensor: MAPE 값 (%)
    """
    abs_percent_error = torch.abs((targets - preds) / (targets + epsilon))
    return torch.mean(abs_percent_error) * 100  # 퍼센트 단위