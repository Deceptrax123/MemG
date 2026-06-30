import torch
from torch import nn


class ManifoldUtils:
    @staticmethod
    def stiefel_regularizer(v_batch):
        v_t_v = torch.bmm(v_batch.transpose(1, 2), v_batch)

        batch_size, dim, rank = v_batch.size()
        I = torch.eye(rank, device=v_batch.device).unsqueeze(
            0).expand(batch_size, -1, -1)
        penalty = torch.norm(v_t_v-I, p='fro', dim=(1, 2))

        return penalty.mean()

    @staticmethod
    def project_to_stiefel(v_batch):
        U, S, Vh = torch.linalg.svd(v_batch, full_matrices=False)
        v_stiefel = torch.bmm(U, Vh)

        return v_stiefel
