import torch
import numpy as np
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer, AutoModel, AutoTokenizer
from agent.layers import *
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
            self.args.output_dim, rank=self.args.rank, eps=self.args.gaussian_eps)

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
                f'sentence-transformers/all-MiniLM-L6-v2')
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

    def volume_regularizer(self, v, d):
        target_log_det = 0
        log_det_d = torch.logdet(d)
        d_inv_sqrt = 1/torch.sqrt(d)

        v_scaled = torch.bmm(d_inv_sqrt, v)

        batch_size, dim, rank = v.size()
        I = torch.eye(rank, device=v.device).unsqueeze(0)

        inner = I+torch.bmm(v_scaled.transpose(1, 2), v_scaled)
        log_det_inner = torch.logdet(inner)

        total_log_det = log_det_d+log_det_inner

        return torch.mean((total_log_det - target_log_det) ** 2)

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

    def diversity_regularizer(self, lambdas, d):

        target_var = 0.5

        penalty_d = F.relu(target_var - d)
        penalty_lambda = F.relu(target_var - lambdas)

        loss_diversity = torch.mean(penalty_d) + torch.mean(penalty_lambda)

        return loss_diversity

    def coefficient(self, mu_q, sigma_q, mu_doc, sigma_doc):
        dist = self.bhattacharya_distance(mu_q, sigma_q, mu_doc, sigma_doc)

        return torch.exp(-dist)

    def _get_logdet(self, v, d):
        logdet_d = torch.logdet(d)
        d_diag = torch.diagonal(d, dim1=-2, dim2=-1)
        inv_sqrt_d = torch.rsqrt(d_diag).unsqueeze(-1)
        v_scaled = v * inv_sqrt_d

        I = torch.eye(self.args.rank, device=v.device).unsqueeze(0)
        inner_core = I + torch.bmm(v_scaled.transpose(1, 2), v_scaled)

        logdet_core = torch.logdet(inner_core)
        return logdet_core + logdet_d

    def hierarchial_consistency(self, q_v, q_d, p_v, p_d):
        vol_q = self._get_logdet(q_v, q_d)
        vol_p = self._get_logdet(p_v, p_d)

        loss = F.relu(vol_q-vol_p+self.args.vol_margin).mean()

        return loss

    def forward(self, query, passage, negative):
        query_mu, q_lambdas, query_v, query_d, query_sigma = self.gaussian_projection(
            query)
        passage_mu, p_lambdas, passage_v, passage_d, passage_sigma = self.gaussian_projection(
            passage)
        negative_mu, _, negative_v, negative_d, negative_sigma = self.gaussian_projection(
            negative)

        inbatch_passage_mu = torch.roll(passage_mu, shifts=1, dims=0)
        inbatch_passage_sigma = torch.roll(passage_sigma, shifts=1, dims=0)

        positive_kl = self.kl_divergence_gaussians(
            query_mu, query_sigma, passage_mu, passage_sigma)
        hard_negative_kl = self.kl_divergence_gaussians(
            query_mu, query_sigma, negative_mu, negative_sigma)
        inbatch_negative_kl = self.kl_divergence_gaussians(
            query_mu, query_sigma, inbatch_passage_mu, inbatch_passage_sigma)

        kl_loss_hard = F.relu(
            positive_kl + self.args.kl_margin - hard_negative_kl).mean()
        kl_loss_inbatch = F.relu(
            positive_kl + self.args.kl_margin - inbatch_negative_kl).mean()
        kl_loss = (kl_loss_hard + kl_loss_inbatch) / 2.0

        positive_bce = self.coefficient(
            query_mu, query_sigma, passage_mu, passage_sigma)
        hard_negative_bce = self.coefficient(
            query_mu, query_sigma, negative_mu, negative_sigma)
        inbatch_negative_bce = self.coefficient(
            query_mu, query_sigma, inbatch_passage_mu, inbatch_passage_sigma)

        pos_bce_loss = -torch.log(positive_bce + 1e-9).mean()
        neg_bce_hard_loss = -torch.log(1 - hard_negative_bce + 1e-9).mean()
        neg_bce_inbatch_loss = - \
            torch.log(1 - inbatch_negative_bce + 1e-9).mean()

        bce_loss = pos_bce_loss + \
            (neg_bce_hard_loss + neg_bce_inbatch_loss) / 2.0

        hierarchy_loss = self.hierarchial_consistency(
            query_v, query_d, passage_v, passage_d)

        reg_loss = hierarchy_loss
        diversity_loss = (self.diversity_regularizer(
            q_lambdas, query_d)+self.diversity_regularizer(p_lambdas, passage_d))/2

        loss = (self.args.kl_weight * kl_loss) + (self.args.bce_weight *
                                                  bce_loss) + (self.args.reg_weight * reg_loss)+(0.05*diversity_loss)

        return loss, kl_loss, bce_loss, reg_loss, query_v, query_d, passage_v, passage_d


# class GaussianEmbeddingTuner(nn.Module):
#     def __init__(self, args):
#         super(GaussianEmbeddingTuner, self).__init__()
#         self.args = args
#         self.model_name = self.args.model_name
#         self.final_dim = self.args.output_dim
#         self.hidden_dim = self.args.hidden_dim

#         self.model_hidden_size = 768
#         self.pretrained_model = self.__load_pretrained__()
#         self.pretrained_model.config.use_cache = False
#         self.z_transform = MLP_Z(
#             self.model_hidden_size, self.hidden_dim, self.args.output_dim)
#         self.center_prediction = MLP_Gaussian_mu(
#             self.args.output_dim, self.args.output_dim)
#         self.decomposed_covariance = MLP_Gaussian_sigma(
#             self.args.output_dim, rank=self.args.rank, eps=self.args.gaussian_eps)

#     def __load_pretrained__(self):

#         print("Loading model...")
#         if self.args.model_name == 'bert-base-uncased':
#             model = BertModel.from_pretrained('/home/models/bert-base-uncased')
#         elif self.args.model_name == 'gte-large-en-v1.5':
#             model = AutoModel.from_pretrained(
#                 'Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True, attn_implementation='eager', dtype=torch.float16)
#         elif self.args.model_name == 'nomic-embed-text-v1.5':
#             model = AutoModel.from_pretrained(
#                 'nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True)
#         elif self.args.model_name == 'all-MiniLM-L6-v2':
#             model = AutoModel.from_pretrained(
#                 f'/home/models/{self.args.model_name}')
#             self.model_hidden_size = 384
#         elif self.args.model_name == 'Qwen3-Embedding-0.6B':
#             model = AutoModel.from_pretrained(
#                 '/home/models/qwen3-embedding-0.6b', torch_dtype=torch.bfloat16)
#             lora_config = LoraConfig(
#                 r=16,
#                 lora_alpha=32,
#                 target_modules=[
#                     "q_proj",
#                     "k_proj",
#                     "v_proj",
#                     "o_proj",
#                     "gate_proj",
#                     "up_proj",
#                     "down_proj"
#                 ],
#                 lora_dropout=0.05,
#                 bias="none",
#                 task_type=TaskType.FEATURE_EXTRACTION
#             )
#             model = get_peft_model(model, lora_config)

#             self.model_hidden_size = 1024

#         print("model loaded !!!!")
#         return model

#     def get_cls(self, encode_inputs):
#         if self.args.model_name == 'bert-base-uncased' or self.args.model_name == 'bert-large-uncased':
#             outputs = self.pretrained_model(**encode_inputs)
#             last_hidden_state = outputs.last_hidden_state
#             input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
#                 -1).expand(last_hidden_state.size()).float()

#             sum_embeddings = torch.sum(
#                 last_hidden_state*input_mask_expanded, 1)
#             sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#             cls_embed = sum_embeddings/sum_mask
#         elif self.args.model_name == 'gte-large-en-v1.5':
#             encode_inputs['attention_mask'] = encode_inputs['attention_mask'].long()
#             outputs = self.pretrained_model(**encode_inputs)
#             cls_embed = outputs.last_hidden_state[:, 0]
#         elif self.args.model_name == 'nomic-embed-text-v1.5':
#             outputs = self.pretrained_model(**encode_inputs)
#             token_embeddings = outputs[0]
#             input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
#                 -1).expand(token_embeddings.size()).float()

#             cls_embed = torch.sum(token_embeddings*input_mask_expanded, 1) / \
#                 torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#             cls_embed = F.normalize(cls_embed, p=2, dim=1)
#         elif self.args.model_name == 'all-MiniLM-L6-v2':
#             outputs = self.pretrained_model(**encode_inputs)
#             token_embeddings = outputs[0]
#             input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
#                 -1).expand(token_embeddings.size()).float()
#             cls_embed = torch.sum(token_embeddings*input_mask_expanded, 1) / \
#                 torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#             cls_embed = F.normalize(cls_embed, p=2, dim=1)
#         elif self.args.model_name == 'Qwen3-Embedding-0.6B':
#             outputs = self.pretrained_model(**encode_inputs)
#             token_embeddings = outputs.last_hidden_state

#             input_mask_expanded = encode_inputs['attention_mask'].unsqueeze(
#                 -1).expand(token_embeddings.size()).float()

#             sum_embeddings = torch.sum(
#                 token_embeddings * input_mask_expanded, 1)
#             sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#             cls_embed = sum_embeddings / sum_mask

#             cls_embed = F.normalize(cls_embed, p=2, dim=1)

#         return cls_embed

#     def gaussian_projection(self, tokenized_inputs):
#         z = self.get_cls(tokenized_inputs)
#         z = self.z_transform(z)
#         z_mu = self.center_prediction(z)
#         z_v, z_d, z_sigma = self.decomposed_covariance(z)

#         return z_mu, z_v, z_d, z_sigma

#     def volume_regularizer(self, v, d):
#         target_log_det = 0
#         log_det_d = torch.logdet(d)
#         d_inv_sqrt = 1/torch.sqrt(d)

#         v_scaled = torch.bmm(d_inv_sqrt, v)

#         batch_size, dim, rank = v.size()
#         I = torch.eye(rank, device=v.device).unsqueeze(0)

#         inner = I+torch.bmm(v_scaled.transpose(1, 2), v_scaled)
#         log_det_inner = torch.logdet(inner)

#         total_log_det = log_det_d+log_det_inner

#         return torch.mean((total_log_det - target_log_det) ** 2)

#     def kl_divergence_gaussians(self, mu_q, sigma_q, mu_d, sigma_d):
#         L_document = torch.linalg.cholesky(sigma_d)
#         solved_sigma = torch.cholesky_solve(sigma_q, L_document)

#         trace_term = torch.diagonal(solved_sigma, dim1=-2, dim2=-1).sum(-1)

#         diff = (mu_d-mu_q).unsqueeze(-1)

#         sol = torch.cholesky_solve(diff, L_document)
#         mahanlobis_term = torch.bmm(diff.transpose(1, 2), sol).squeeze()

#         log_det_doc = 2 * \
#             torch.sum(torch.log(torch.diagonal(
#                 L_document, dim1=-2, dim2=-1)), dim=-1)
#         L_query = torch.linalg.cholesky(sigma_q)
#         log_det_query = 2 * \
#             torch.sum(torch.log(torch.diagonal(
#                 L_query, dim1=-2, dim2=-1)), dim=-1)

#         term_logdet = log_det_doc-log_det_query

#         kl = 0.5*(trace_term+mahanlobis_term-self.final_dim+term_logdet)

#         return kl

#     def bhattacharya_distance(self, mu1, sigma1, mu2, sigma2):
#         sigma_avg = 0.5 * (sigma1 + sigma2)

#         diff = (mu1 - mu2).unsqueeze(-1)

#         L_avg = torch.linalg.cholesky(sigma_avg)
#         sol = torch.cholesky_solve(diff, L_avg)

#         term1 = 0.125 * torch.bmm(diff.transpose(1, 2), sol).squeeze()

#         log_det_avg = 2 * \
#             torch.sum(torch.log(torch.diagonal(
#                 L_avg, dim1=-2, dim2=-1)), dim=-1)

#         L1 = torch.linalg.cholesky(sigma1)
#         log_det_1 = 2 * \
#             torch.sum(torch.log(torch.diagonal(L1, dim1=-2, dim2=-1)), dim=-1)

#         L2 = torch.linalg.cholesky(sigma2)
#         log_det_2 = 2 * \
#             torch.sum(torch.log(torch.diagonal(L2, dim1=-2, dim2=-1)), dim=-1)

#         term2 = 0.5 * (log_det_avg - 0.5 * (log_det_1 + log_det_2))

#         return term1 + term2

#     def coefficient(self, mu_q, sigma_q, mu_doc, sigma_doc):
#         dist = self.bhattacharya_distance(mu_q, sigma_q, mu_doc, sigma_doc)

#         return torch.exp(-dist)

#     def _get_logdet(self, v, d):
#         logdet_d = torch.logdet(d)
#         d_diag = torch.diagonal(d, dim1=-2, dim2=-1)
#         inv_sqrt_d = torch.rsqrt(d_diag).unsqueeze(-1)
#         v_scaled = v * inv_sqrt_d

#         I = torch.eye(self.args.rank, device=v.device).unsqueeze(0)
#         inner_core = I + torch.bmm(v_scaled.transpose(1, 2), v_scaled)

#         logdet_core = torch.logdet(inner_core)
#         return logdet_core + logdet_d

#     def hierarchial_consistency(self, q_v, q_d, p_v, p_d):
#         vol_q = self._get_logdet(q_v, q_d)
#         vol_p = self._get_logdet(p_v, p_d)

#         loss = F.relu(vol_q-vol_p+self.args.vol_margin).mean()

#         return loss

#     def forward(self, query, passage, negative):
#         query_mu, query_v, query_d, query_sigma = self.gaussian_projection(
#             query)
#         passage_mu, passage_v, passage_d, passage_sigma = self.gaussian_projection(
#             passage)
#         negative_mu, negative_v, negative_d, negative_sigma = self.gaussian_projection(
#             negative)

#         inbatch_passage_mu = torch.roll(passage_mu, shifts=1, dims=0)
#         inbatch_passage_sigma = torch.roll(passage_sigma, shifts=1, dims=0)

#         positive_kl = self.kl_divergence_gaussians(
#             query_mu, query_sigma, passage_mu, passage_sigma)
#         hard_negative_kl = self.kl_divergence_gaussians(
#             query_mu, query_sigma, negative_mu, negative_sigma)
#         inbatch_negative_kl = self.kl_divergence_gaussians(
#             query_mu, query_sigma, inbatch_passage_mu, inbatch_passage_sigma)

#         kl_loss_hard = F.relu(
#             positive_kl + self.args.kl_margin - hard_negative_kl).mean()
#         kl_loss_inbatch = F.relu(
#             positive_kl + self.args.kl_margin - inbatch_negative_kl).mean()
#         kl_loss = (kl_loss_hard + kl_loss_inbatch) / 2.0

#         positive_bce = self.coefficient(
#             query_mu, query_sigma, passage_mu, passage_sigma)
#         hard_negative_bce = self.coefficient(
#             query_mu, query_sigma, negative_mu, negative_sigma)
#         inbatch_negative_bce = self.coefficient(
#             query_mu, query_sigma, inbatch_passage_mu, inbatch_passage_sigma)

#         pos_bce_loss = -torch.log(positive_bce + 1e-9).mean()
#         neg_bce_hard_loss = -torch.log(1 - hard_negative_bce + 1e-9).mean()
#         neg_bce_inbatch_loss = - \
#             torch.log(1 - inbatch_negative_bce + 1e-9).mean()

#         bce_loss = pos_bce_loss + \
#             (neg_bce_hard_loss + neg_bce_inbatch_loss) / 2.0

#         # query_reg = self.volume_regularizer(query_v, query_d)
#         # passage_reg = self.volume_regularizer(passage_v, passage_d)
#         # negative_reg = self.volume_regularizer(negative_v, negative_d)
#         # stability_loss = query_reg + passage_reg
#         hierarchy_loss = self.hierarchial_consistency(
#             query_v, query_d, passage_v, passage_d)

#         reg_loss = hierarchy_loss

#         loss = (self.args.kl_weight * kl_loss) + \
#                (self.args.bce_weight * bce_loss) + \
#                (self.args.reg_weight * reg_loss)

#         return loss, kl_loss, bce_loss, reg_loss, query_v, query_d, passage_v, passage_d
