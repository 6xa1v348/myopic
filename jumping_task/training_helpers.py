import numpy as np
import torch
import torch.nn.functional as F


def metric_fixed_point(cost_matrix, gamma=1.0, eps=1e-9):
    if gamma == 0: return cost_matrix
    n, m = cost_matrix.shape
    d_metric = np.zeros_like(cost_matrix)
    def fixed_point_operator(d_metric):
        d_metric_new = np.empty_like(d_metric)
        for i in range(n):
            for j in range(m):
                d_metric_new[i, j] = cost_matrix[i, j] + \
                    gamma * d_metric[min(i + 1, n - 1), min(j + 1, m - 1)]
        return d_metric_new
    
    while True:
        d_metric_new = fixed_point_operator(d_metric)
        if np.sum(np.abs(d_metric - d_metric_new)) < eps:
            break
        else:
            d_metric = d_metric_new
    return d_metric


def calculate_action_cost_matrix(actions_1, actions_2):
    action_equality = np.equal(
        np.expand_dims(actions_1, axis=1), np.expand_dims(actions_2, axis=0)
    )
    return 1. - action_equality.astype(np.float32)


def cosine_similarity(x, y):
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return torch.mm(x_norm, y_norm.T)


# Contrastive Metric Enbeddings / SimCLR
def soft_simclr_loss(similarity_matrix,
                     metric_values,
                     temperature=1.0,
                     soft_coupling_temperature=1.0,
                     use_coupling_weights=True):
    assert similarity_matrix.shape == metric_values.shape
    similarity_matrix = similarity_matrix / temperature
    def compute_loss(sim, d, eps=1e-9):
        index = torch.argmin(d, dim=1, keepdim=True)
        mask = torch.zeros_like(d)
        mask[torch.arange(mask.size(0)).unsqueeze(1), index] = 1
        positive = sim[mask.bool()].view(mask.size(0), -1)
        negative = sim
        if use_coupling_weights:
            d /= soft_coupling_temperature
            coupling = torch.exp(-d)
            positive_weights = -d[mask.bool()].view(mask.size(0), -1)
            positive = positive + positive_weights
            negative_weights = torch.log(1.0 - coupling + eps)
            negative_weights = -d.masked_fill((1 - mask).bool(), 0) + \
                negative_weights.masked_fill(mask.bool(), 0)
            negative = negative + negative_weights
        negative = torch.logsumexp(negative, dim=1, keepdim=True)
        loss = torch.mean(negative - positive)
        return loss
    
    loss1 = compute_loss(similarity_matrix, metric_values)
    loss2 = compute_loss(similarity_matrix.T, metric_values.T)
    return loss1 + loss2


def multi_soft_simclr_loss(similarity_matrix,
                           metric_values,
                           temperature=1.0,
                           soft_coupling_temperature=1.0):
    assert similarity_matrix.shape == metric_values.shape
    m, n = similarity_matrix.shape
    similarity_matrix = similarity_matrix / temperature
    def compute_loss(sim, d, eps=1e-9):
        mask = (1 - d).bool()
        coupling = torch.exp(-d / soft_coupling_temperature)
        positive = coupling * torch.exp(sim)
        negative = (1 - coupling) * torch.exp(sim)
        negatives = torch.sum(negative, dim=1).unsqueeze(1).expand(-1, n)
        diff = (positive - negative) * mask
        denom = negatives + diff
        probs = torch.masked_select(positive / denom, mask)
        loss = -torch.log(probs + eps)
        return torch.mean(loss)
    
    loss1 = compute_loss(similarity_matrix, metric_values)
    loss2 = compute_loss(similarity_matrix.T, metric_values.T)
    return loss1 + loss2


def cross_entropy_loss(model, obs, acs):
    pi = model.actor(obs)
    loss = F.cross_entropy(pi, target=acs)
    return loss


def negative_loglikelihood_loss(model, obs, acs, next_obs):
    h = model.actor.encoder(obs)
    mu = model.transition_model(h, acs)
    next_h = model.actor.encoder(next_obs)
    diff = (mu - next_h.detach())
    loss = torch.mean(0.5 * diff.pow(2))
    return loss
