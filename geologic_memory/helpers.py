from transformers import BertTokenizer, AutoTokenizer
from geologic_memory.model import GaussianEmbeddingTuner
import torch


def bhattacharya_coefficient(mu1, v1, d1, mu2, v2, d2):
    D = mu1.shape[0]

    eps = 1e-5
    sigma1 = torch.matmul(v1, v1.T) + torch.diag_embed(d1) + \
        (eps * torch.eye(D, device=mu1.device))
    sigma2 = torch.matmul(v2, v2.T)+torch.diag_embed(d2) + \
        (eps*torch.eye(D, device=mu2.device))

    sigma_avg = (sigma1+sigma2)/2
    mu_diff = (mu1-mu2).unsqueeze(1)

    inv_sigma_diff = torch.linalg.solve(sigma_avg, mu_diff)
    term1 = 0.125*torch.matmul(mu_diff.T, inv_sigma_diff).squeeze()

    _, logdet_avg = torch.linalg.slogdet(sigma_avg)
    _, logdet_1 = torch.linalg.slogdet(sigma1)
    _, logdet_2 = torch.linalg.slogdet(sigma2)

    term2 = 0.5*(logdet_avg-0.5*(logdet_1+logdet_2))

    Db = term1+term2

    Db = torch.clamp(Db, min=0.0, max=50.0)
    Bc = torch.exp(-Db)

    return Bc.item()


def build_belief_graph(retrieved_nodes, args):
    edges = []
    num_nodes = len(retrieved_nodes)

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            node_A, node_B = retrieved_nodes[i], retrieved_nodes[j]

            bc_score = bhattacharya_coefficient(
                node_A.mu, node_A.V, node_A.log_d, node_B.mu, node_B.V, node_B.log_d)
            set_a, set_b = set(node_A.entities), set(node_B.entities)
            intersection = len(set_a.intersection(set_b))
            entity_score = (intersection/len(set_a.union(set_b))
                            ) if set_a or set_b else 0.0

            if intersection > 0:
                entity_score = max(entity_score, 0.8)

            topic_score = 1.0 if node_A.topic == node_B.topic else 0

            total_weight = (args.bc_weight*bc_score) + (args.entity_weight *
                                                        entity_score)+(args.topic_weight*topic_score)

            if total_weight >= args.base_threshold:
                edges.append((i, j, total_weight))

    return edges


def load_tokenizer(args):
    if args.model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(
            '/home/models/bert-base-uncased')
    elif args.model_name == 'gte-large-en-v1.5':
        tokenizer = AutoTokenizer.from_pretrained(
            'Alibaba-NLP/gte-large-en-v1.5')
    elif args.model_name == 'nomic-embed-text-v1.5':
        tokenizer = AutoTokenizer.from_pretrained(
            '/home/models/bert-base-uncased')
    elif args.model_name == 'all-MiniLM-L6-v2':
        tokenizer = AutoTokenizer.from_pretrained(
            f'sentence-transformers/all-MiniLM-L6-v2')
    elif args.model_name == 'Qwen3-Embedding-0.6B':
        tokenizer = AutoTokenizer.from_pretrained(
            '/home/models/qwen3-embedding-0.6b', padding_side='left')
    elif args.model_name == 'bge-base-en-v1.5':
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return tokenizer


def load_model(args, weights_path):
    model = GaussianEmbeddingTuner(args)
    model.load_state_dict(torch.load(weights_path, map_location=args.device, weights_only=True))

    return model
