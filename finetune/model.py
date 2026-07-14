import torch
import numpy as np
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer, AutoModel, AutoTokenizer
try:
    from finetune.layers import *
except ModuleNotFoundError:
    from layers import *
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model


class GaussianEmbeddingTuner(nn.Module):
    def __init__(self, args):
        super(GaussianEmbeddingTuner, self).__init__()
        self.args = args
        self.model_name = self.args.model_name
        self.final_dim = self.args.output_dim
        self.hidden_dim = self.args.hidden_dim

        self.model_hidden_size = 768
        self.pretrained_model = self.__load_pretrained__()
        self.pretrained_model.config.use_cache = False
        self.z_transform = MLP_Z(
            self.model_hidden_size, self.hidden_dim, self.args.output_dim)
        self.center_prediction = MLP_Gaussian_mu(
            self.args.output_dim, self.args.output_dim)
        self.decomposed_covariance = MLP_Gaussian_Stiefel_Sigma(
            self.args.output_dim,
            rank=self.args.rank,
            eps=self.args.gaussian_eps,
            decomposition_type=getattr(
                self.args, 'decomposition_type', 'stiefel_projected_lambdas'),
            stiefel_iterations=getattr(self.args, 'stiefel_iterations', 4),
            min_variance=getattr(self.args, 'min_variance', 1e-5),
            max_variance=getattr(self.args, 'max_variance', 10.0),
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        legacy_keys = {
            'decomposed_covariance.d_proj.weight',
            'decomposed_covariance.d_proj.bias',
        }
        if any(key in state_dict for key in legacy_keys):
            state_dict = state_dict.copy()
            for key in legacy_keys:
                state_dict.pop(key, None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def __load_pretrained__(self):

        print("Loading model...")
        if self.args.model_name == 'bert-base-uncased':
            model = BertModel.from_pretrained('/home/models/bert-base-uncased')
        elif self.args.model_name == 'gte-large-en-v1.5':
            model = AutoModel.from_pretrained(
                'Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, attn_implementation='eager', dtype=torch.float16)
        elif self.args.model_name == 'nomic-embed-text-v1.5':
            model = AutoModel.from_pretrained(
                'nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
        elif self.args.model_name == 'all-MiniLM-L6-v2':
            model = AutoModel.from_pretrained(
                f'sentence-transformers/{self.args.model_name}')
            self.model_hidden_size = 384
        elif self.args.model_name == 'bge-base-en-v1.5':
            model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
            self.model_hidden_size = 768
        elif self.args.model_name == 'Qwen3-Embedding-0.6B':
            model = AutoModel.from_pretrained(
                '/home/models/qwen3-embedding-0.6b', torch_dtype=torch.bfloat16)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj"
                ],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            model = get_peft_model(model, lora_config)
            self.model_hidden_size = 1024

        print("model loaded !!!!")
        return model

    def get_cls(self, encode_inputs):
        if self.args.model_name == 'bert-base-uncased' or self.args.model_name == 'bert-large-uncased':
            outputs = self.pretrained_model(**encode_inputs)
            last_hidden_state = outputs.last_hidden_state
            input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
                -1).expand(last_hidden_state.size()).float()

            sum_embeddings = torch.sum(
                last_hidden_state*input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            cls_embed = sum_embeddings/sum_mask
        elif self.args.model_name == 'gte-large-en-v1.5':
            encode_inputs['attention_mask'] = encode_inputs['attention_mask'].long()
            outputs = self.pretrained_model(**encode_inputs)
            cls_embed = outputs.last_hidden_state[:, 0]
        elif self.args.model_name == 'nomic-embed-text-v1.5':
            outputs = self.pretrained_model(**encode_inputs)
            token_embeddings = outputs[0]
            input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
                -1).expand(token_embeddings.size()).float()

            cls_embed = torch.sum(token_embeddings*input_mask_expanded, 1) / \
                torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            cls_embed = F.normalize(cls_embed, p=2, dim=1)
        elif self.args.model_name == 'all-MiniLM-L6-v2':
            outputs = self.pretrained_model(**encode_inputs)
            token_embeddings = outputs[0]
            input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
                -1).expand(token_embeddings.size()).float()
            cls_embed = torch.sum(token_embeddings*input_mask_expanded, 1) / \
                torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            cls_embed = F.normalize(cls_embed, p=2, dim=1)
        elif self.args.model_name == 'Qwen3-Embedding-0.6B':
            outputs = self.pretrained_model(**encode_inputs)
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
                -1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            cls_embed = sum_embeddings / sum_mask

            cls_embed = F.normalize(cls_embed, p=2, dim=1)
        elif self.args.model_name == 'bge-base-en-v1.5':
            outputs = self.pretrained_model(**encode_inputs)
            embeddings = outputs[0][:, 0]

            cls_embed = F.normalize(embeddings, p=2, dim=1)

        return cls_embed

    def gaussian_projection(self, tokenized_inputs):
        z = self.get_cls(tokenized_inputs)
        z = self.z_transform(z)
        z_mu = self.center_prediction(z)
        z_v, lambdas, z_d, z_sigma = self.decomposed_covariance(z)

        return z_mu, lambdas, z_v, z_d, z_sigma

    def kl_divergence_gaussians(self, mu_q, sigma_q, mu_d, sigma_d):
        L_document = torch.linalg.cholesky(sigma_d)
        solved_sigma = torch.cholesky_solve(sigma_q, L_document)

        trace_term = torch.diagonal(solved_sigma, dim1=-2, dim2=-1).sum(-1)

        diff = (mu_d-mu_q).unsqueeze(-1)

        sol = torch.cholesky_solve(diff, L_document)
        mahanlobis_term = torch.bmm(diff.transpose(1, 2), sol).squeeze()

        log_det_doc = 2 * torch.sum(torch.log(torch.diagonal(
            L_document, dim1=-2, dim2=-1)), dim=-1)
        L_query = torch.linalg.cholesky(sigma_q)
        log_det_query = 2 * torch.sum(torch.log(torch.diagonal(
            L_query, dim1=-2, dim2=-1)), dim=-1)

        term_logdet = log_det_doc-log_det_query

        kl = 0.5*(trace_term+mahanlobis_term-self.final_dim+term_logdet)

        return kl

    def bhattacharya_distance(self, mu1, sigma1, mu2, sigma2):
        sigma_avg = 0.5 * (sigma1 + sigma2)

        diff = (mu1 - mu2).unsqueeze(-1)

        L_avg = torch.linalg.cholesky(sigma_avg)
        sol = torch.cholesky_solve(diff, L_avg)

        term1 = 0.125 * torch.bmm(diff.transpose(1, 2), sol).squeeze()

        log_det_avg = 2 * torch.sum(torch.log(torch.diagonal(
            L_avg, dim1=-2, dim2=-1)), dim=-1)

        L1 = torch.linalg.cholesky(sigma1)
        log_det_1 = 2 * \
            torch.sum(torch.log(torch.diagonal(L1, dim1=-2, dim2=-1)), dim=-1)

        L2 = torch.linalg.cholesky(sigma2)
        log_det_2 = 2 * \
            torch.sum(torch.log(torch.diagonal(L2, dim1=-2, dim2=-1)), dim=-1)

        term2 = 0.5 * (log_det_avg - 0.5 * (log_det_1 + log_det_2))

        return term1 + term2

    def coefficient(self, mu_q, sigma_q, mu_doc, sigma_doc):
        dist = self.bhattacharya_distance(mu_q, sigma_q, mu_doc, sigma_doc)

        return torch.exp(-dist)

    def _get_logdet(self, V, lambdas, d):
        d_diag = torch.diagonal(d, dim1=-2, dim2=-1)
        logdet_d = torch.sum(
            torch.log(torch.diagonal(d, dim1=-2, dim2=-1)), dim=-1)
        d_inv_sqrt = torch.rsqrt(d_diag).unsqueeze(-1)
        lambda_sqrt = torch.sqrt(lambdas).unsqueeze(1)

        V_scaled = V * d_inv_sqrt * lambda_sqrt
        inner_core = torch.bmm(V_scaled.transpose(1, 2), V_scaled)

        rank = lambdas.size(-1)
        I = torch.eye(rank, device=V.device).unsqueeze(
            0).expand(V.size(0), -1, -1)
        core_matrix = I + inner_core

        logdet_core = torch.logdet(core_matrix)
        total_logdet = logdet_core+logdet_d

        return total_logdet

    def hierarchial_consistency(self, q_v, q_lambdas, q_d, p_v, p_lambdas, p_d):
        vol_q = self._get_logdet(q_v, q_lambdas, q_d)
        vol_p = self._get_logdet(p_v, p_lambdas, p_d)

        loss = F.relu(vol_q-vol_p+self.args.vol_margin).mean()

        return loss

    def exact_bures_wasserstein(self, mu_q, V_q, lambdas_q, d_q, mu_p, V_p, lambdas_p, d_p):
        batch_size, dim, rank = V_q.size()
        eps = self.args.gaussian_eps

        mean_dist = torch.sum((mu_q - mu_p)**2, dim=-1)

        trace_q = torch.sum(lambdas_q, dim=-1) + torch.sum(d_q, dim=-1)
        trace_p = torch.sum(lambdas_p, dim=-1) + torch.sum(d_p, dim=-1)

        diag_cross_term = torch.sum(
            torch.sqrt(torch.clamp(d_q * d_p, min=eps * eps)), dim=-1
        )

        Vq_T_Vp = torch.bmm(V_q.transpose(1, 2), V_p)

        sqrt_lambdas_q = torch.sqrt(
            torch.clamp(lambdas_q, min=eps)).unsqueeze(-1)
        sqrt_lambdas_p = torch.sqrt(
            torch.clamp(lambdas_p, min=eps)).unsqueeze(1)

        scaled_Vq_T_Vp = sqrt_lambdas_q * Vq_T_Vp * sqrt_lambdas_p

        _, S, _ = torch.linalg.svd(scaled_Vq_T_Vp)

        subspace_cross_term = torch.sum(S, dim=-1)

        w2_dist = mean_dist + trace_q + trace_p - 2 * \
            (diag_cross_term + subspace_cross_term)

        return torch.clamp(w2_dist, min=0.0)

    def lambda_anti_collapse_regularizer(self, lambdas):

        target_min_var = 0.5

        lambda_deficit = F.relu(target_min_var - lambdas)

        loss_collapse_lambda = torch.mean(lambda_deficit ** 2)

        return loss_collapse_lambda

    def eigenvalue_spread_regularizer(self, lambdas):
        lambda_variance = torch.var(lambdas, dim=-1, unbiased=False)

        target_spread = 1.0

        spread_deficit = F.relu(target_spread - lambda_variance)

        loss_spread = torch.mean(spread_deficit ** 2)

        return loss_spread

    def forward(self, query_inputs, passage_inputs, negative_inputs):
        query_mu, q_lambdas, query_v, query_d_matrix, _ = self.gaussian_projection(
            query_inputs)
        passage_mu, p_lambdas, passage_v, passage_d_matrix, _ = self.gaussian_projection(
            passage_inputs)
        negative_mu, n_lambdas, negative_v, negative_d_matrix, _ = self.gaussian_projection(
            negative_inputs)

        batch_size, dim = query_mu.size()
        rank = query_v.size(2)
        num_hard_negs = negative_mu.size(0) // batch_size

        query_d = torch.diagonal(query_d_matrix, dim1=-2, dim2=-1)
        passage_d = torch.diagonal(passage_d_matrix, dim1=-2, dim2=-1)
        negative_d = torch.diagonal(negative_d_matrix, dim1=-2, dim2=-1)

        q_mu_exp = query_mu.unsqueeze(
            1).expand(-1, batch_size, -1).reshape(-1, dim)
        q_v_exp = query_v.unsqueeze(
            1).expand(-1, batch_size, -1, -1).reshape(-1, dim, rank)
        q_lam_exp = q_lambdas.unsqueeze(
            1).expand(-1, batch_size, -1).reshape(-1, rank)
        q_d_exp = query_d.unsqueeze(
            1).expand(-1, batch_size, -1).reshape(-1, dim)

        p_mu_exp = passage_mu.unsqueeze(0).expand(
            batch_size, -1, -1).reshape(-1, dim)
        p_v_exp = passage_v.unsqueeze(0).expand(
            batch_size, -1, -1, -1).reshape(-1, dim, rank)
        p_lam_exp = p_lambdas.unsqueeze(0).expand(
            batch_size, -1, -1).reshape(-1, rank)
        p_d_exp = passage_d.unsqueeze(0).expand(
            batch_size, -1, -1).reshape(-1, dim)

        flat_inbatch_dists = self.exact_bures_wasserstein(
            q_mu_exp, q_v_exp, q_lam_exp, q_d_exp,
            p_mu_exp, p_v_exp, p_lam_exp, p_d_exp
        )

        inbatch_dist_matrix = flat_inbatch_dists.view(batch_size, batch_size)

        neg_mu_reshaped = negative_mu.view(batch_size, num_hard_negs, -1)
        neg_v_reshaped = negative_v.view(batch_size, num_hard_negs, dim, rank)
        neg_lam_reshaped = n_lambdas.view(batch_size, num_hard_negs, -1)
        neg_d_reshaped = negative_d.view(batch_size, num_hard_negs, -1)

        hard_neg_dist_matrix = torch.zeros(
            batch_size, num_hard_negs, device=query_mu.device)

        for k in range(num_hard_negs):
            k_dists = self.exact_bures_wasserstein(
                query_mu, query_v, q_lambdas, query_d,
                neg_mu_reshaped[:, k], neg_v_reshaped[:,
                                                      k], neg_lam_reshaped[:, k], neg_d_reshaped[:, k]
            )
            hard_neg_dist_matrix[:, k] = k_dists

        all_dists = torch.cat(
            [inbatch_dist_matrix, hard_neg_dist_matrix], dim=1)

        tau = 0.05
        logits = -all_dists / tau

        labels = torch.arange(batch_size, device=query_mu.device)

        contrastive_loss = F.cross_entropy(logits, labels)

        hierarchy_loss = self.hierarchial_consistency(
            query_v, q_lambdas, query_d_matrix, passage_v, p_lambdas, passage_d_matrix
        )

        var_loss_q = self.lambda_anti_collapse_regularizer(q_lambdas)
        var_loss_p = self.lambda_anti_collapse_regularizer(p_lambdas)
        var_loss_n = self.lambda_anti_collapse_regularizer(n_lambdas)

        variance_loss = (var_loss_q + var_loss_p + var_loss_n) / 3.0

        spread_loss = (self.eigenvalue_spread_regularizer(
            q_lambdas)+self.eigenvalue_spread_regularizer(p_lambdas))/2

        loss = (self.args.bce_weight * contrastive_loss) + 0.05*variance_loss +\
            self.args.reg_weight*hierarchy_loss + 0.1*spread_loss

        return loss, contrastive_loss, contrastive_loss, hierarchy_loss, query_v, q_lambdas, query_d_matrix, passage_v, p_lambdas, passage_d_matrix

    # def forward(self, query, passage, negative):
    #     query_mu, q_lambdas, query_v, query_d, query_sigma = self.gaussian_projection(
    #         query)
    #     passage_mu, p_lambdas, passage_v, passage_d, passage_sigma = self.gaussian_projection(
    #         passage)
    #     negative_mu, _, negative_v, negative_d, negative_sigma = self.gaussian_projection(
    #         negative)

    #     inbatch_passage_mu = torch.roll(passage_mu, shifts=1, dims=0)
    #     inbatch_passage_sigma = torch.roll(passage_sigma, shifts=1, dims=0)

    #     positive_kl = self.kl_divergence_gaussians(
    #         query_mu, query_sigma, passage_mu, passage_sigma)
    #     hard_negative_kl = self.kl_divergence_gaussians(
    #         query_mu, query_sigma, negative_mu, negative_sigma)
    #     inbatch_negative_kl = self.kl_divergence_gaussians(
    #         query_mu, query_sigma, inbatch_passage_mu, inbatch_passage_sigma)

    #     kl_loss_hard = F.relu(
    #         positive_kl + self.args.kl_margin - hard_negative_kl).mean()
    #     kl_loss_inbatch = F.relu(
    #         positive_kl + self.args.kl_margin - inbatch_negative_kl).mean()
    #     kl_loss = (kl_loss_hard + kl_loss_inbatch) / 2.0

    #     positive_bce = self.coefficient(
    #         query_mu, query_sigma, passage_mu, passage_sigma)
    #     hard_negative_bce = self.coefficient(
    #         query_mu, query_sigma, negative_mu, negative_sigma)
    #     inbatch_negative_bce = self.coefficient(
    #         query_mu, query_sigma, inbatch_passage_mu, inbatch_passage_sigma)

    #     pos_bce_loss = -torch.log(positive_bce + 1e-9).mean()
    #     neg_bce_hard_loss = -torch.log(1 - hard_negative_bce + 1e-9).mean()
    #     neg_bce_inbatch_loss = - \
    #         torch.log(1 - inbatch_negative_bce + 1e-9).mean()

    #     bce_loss = pos_bce_loss + \
    #         (neg_bce_hard_loss + neg_bce_inbatch_loss) / 2.0

    #     hierarchy_loss = self.hierarchial_consistency(
    #         query_v, query_d, passage_v, passage_d)

    #     reg_loss = hierarchy_loss
    #     diversity_loss = (self.diversity_regularizer(
    #         q_lambdas, query_d)+self.diversity_regularizer(p_lambdas, passage_d))/2

    #     loss = (self.args.kl_weight * kl_loss) + (self.args.bce_weight *
    #                                               bce_loss) + (self.args.reg_weight * reg_loss)+(0.05*diversity_loss)

    #     return loss, kl_loss, bce_loss, reg_loss, query_v, query_d, passage_v, passage_d
