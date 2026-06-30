import torch
import torch.nn as nn
from optimizer import RiemannianCayleySGD
from math_utils import *


class AGMRiemmannianPolicy:
    def __init__(self, alpha_success=1.0, beta_consistency=0.8, gamma_inertia=0.2):
        self.alpha = alpha_success
        self.beta = beta_consistency
        self.gamma = gamma_inertia

    def execute_rollout(self, query_node, subgraph_nodes, subgraph_edges, steps=30, lr=0.05):
        action_params = []
        for node in subgraph_nodes:
            action_params.extend([node.mu, node.V, node.log_lam, node.log_d])

        optimizer = RiemannianCayleySGD(action_params, lr=lr)

        q_mu_raw, q_V_raw, q_lam_raw, q_d_raw = query_node.get_params()
        q_mu = q_mu_raw.detach()
        q_V = q_V_raw.detach()
        q_lam = q_lam_raw.detach()
        q_d = q_d_raw.detach()

        print(
            f"Running geometric belief revision on {len(subgraph_nodes)} nodes...")
        for step in range(steps):
            optimizer.zero_grad()
            loss_success = torch.tensor(
                0.0, device=q_mu.device, requires_grad=True).clone()
            loss_consistency = torch.tensor(
                0.0, device=q_mu.device, requires_grad=True).clone()
            loss_inertia = torch.tensor(
                0.0, device=q_mu.device, requires_grad=True).clone()

            for node in subgraph_nodes:
                n_mu, n_V, n_lam, n_d = node.get_params()
                loss_success += GaussianMath.geometric_energy(
                    q_mu, q_V, q_lam, q_d, n_mu, n_V, n_lam, n_d
                )
                loss_inertia += torch.norm(n_mu - node.mu_orig.detach())**2

            for src_idx, tgt_idx, weight in subgraph_edges:
                src, tgt = subgraph_nodes[src_idx], subgraph_nodes[tgt_idx]
                s_mu, s_V, s_lam, s_d = src.get_params()
                t_mu, t_V, t_lam, t_d = tgt.get_params()

                dist = GaussianMath.geometric_energy(
                    s_mu, s_V, s_lam, s_d, t_mu, t_V, t_lam, t_d)
                loss_consistency += weight * dist

            total_energy = (self.alpha * loss_success) + \
                           (self.beta * loss_consistency) + \
                           (self.gamma * loss_inertia)

            total_energy.backward()
            optimizer.step()

        return subgraph_nodes
