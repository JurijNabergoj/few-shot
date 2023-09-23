import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable


def relation_net_episode(model: Module,
                         optimiser: Optimizer,
                         loss_fn: Callable,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         n_shot: int,
                         k_way: int,
                         q_queries: int,
                         train: bool):
    """Performs a single training episode for a Relation Network.

    # Arguments
        model: Consists of encoding and relation models to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update

    # Returns
        loss: Loss of the Relation Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """

    if train:
        # Zero gradients
        model.encoder.train()
        model.relation.train()
        optimiser.zero_grad()
    else:
        model.encoder.eval()
        model.relation.eval()

    # Embed all samples

    support_labels = y[:n_shot * k_way]
    query_labels = y[n_shot * k_way:]
    embeddings = model.encoder(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]

    relations = compute_relations(model, support, queries, k_way, n_shot, q_queries)

    one_hot_labels = torch.zeros(q_queries * k_way, k_way).to('cuda').scatter_(1, query_labels.view(-1, 1), 1)
    loss = loss_fn(relations.float(), one_hot_labels.float())

    y_pred = relations.data

    if train:
        # Take gradient step
        loss.backward()
        clip_grad_norm_(model.encoder.parameters(), 0.5)
        clip_grad_norm_(model.relation.parameters(), 0.5)
        optimiser.step()
    else:
        pass

    return loss, y_pred


def compute_relations(model: Module, support: torch.Tensor, query: torch.Tensor, k: int, n: int,
                      q: int) -> torch.Tensor:
    """Compute class relations from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task

    # Returns
        class_relations: Relations aka outputs of relation CNN for each class
    """

    support = support.view(k, n, support.shape[1], support.shape[2], support.shape[3])
    support = torch.sum(support, 1).squeeze(1)

    support_ext = support.unsqueeze(0).repeat(q * k, 1, 1, 1, 1)
    queries_ext = torch.transpose(query.unsqueeze(0).repeat(k, 1, 1, 1, 1), 0, 1)
    relation_pairs = torch.cat((support_ext, queries_ext), 2)
    relation_pairs = relation_pairs.view(-1, relation_pairs.shape[2], relation_pairs.shape[3], relation_pairs.shape[4])
    relations = model.relation(relation_pairs).view(-1, k)

    return relations
