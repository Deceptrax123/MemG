import torch
import torch.nn.functional as F
from rapidfuzz import fuzz


class GaussianMath:
    @staticmethod
    def relaxed_wassertian_energy(mu1, v1, d1, mu2, v2, d2):
        mean_diff = torch.norm(mu1-mu2)**2

        trace_1 = torch.trace(v1@v1.T)+torch.sum(d1)
        trace_2 = torch.trace(v2@v2.T)+torch.sum(d2)

        cross_term = 2*torch.sqrt(torch.clamp(trace_1*trace_2, min=1e-6))

        return mean_diff+trace_2+trace_1-cross_term

    @staticmethod
    def compute_covariance(V, d):
        return (V @ V.T) + torch.diag(d)

    @staticmethod
    def geometric_energy(mu1, V1, d1, mu2, V2, d2):
        mean_diff = torch.norm(mu1 - mu2)**2

        cov1 = GaussianMath.compute_covariance(V1, d1)
        cov2 = GaussianMath.compute_covariance(V2, d2)

        cov_diff = torch.norm(cov1 - cov2, p='fro')**2

        return mean_diff + (0.5 * cov_diff)


class FastGeometricResolver:
    @staticmethod
    def compute_overlap(mu1, V1, d1, mu2, V2, d2):
        d_vec1 = d1 if d1.dim() == 1 else torch.diagonal(d1)
        d_vec2 = d2 if d2.dim() == 1 else torch.diagonal(d2)

        sigma1 = d_vec1 + torch.sum(V1**2, dim=-1)
        sigma2 = d_vec2 + torch.sum(V2**2, dim=-1)

        mean_dist = torch.sum((mu1 - mu2)**2)
        var_dist = torch.sum((torch.sqrt(sigma1) - torch.sqrt(sigma2))**2)

        total_dist = mean_dist + var_dist

        overlap = torch.exp(-total_dist)
        return overlap.item()

    @staticmethod
    def build_and_resolve_graph(retrieved_nodes, overlap_threshold=0.4, topic_sim_threshold=0.85):
        edges = []
        num_nodes = len(retrieved_nodes)

        for node in retrieved_nodes:
            node.status = "UNCERTAIN"
            node._newer_updates = 0
            node._corroborators = 0

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                node_A = retrieved_nodes[i]
                node_B = retrieved_nodes[j]

                set_A = set(e.lower() for e in node_A.entities)
                set_B = set(e.lower() for e in node_B.entities)
                shared_entities = len(set_A.intersection(set_B)) > 0

                topic_sim = torch.dot(
                    node_A.topic_embedding, node_B.topic_embedding).item()
                shared_topic = (topic_sim > topic_sim_threshold)

                if not (shared_entities or shared_topic):
                    continue

                A_mu, A_V, _, A_d = node_A.get_params()
                B_mu, B_V, _, B_d = node_B.get_params()

                overlap_score = FastGeometricResolver.compute_overlap(
                    A_mu, A_V, A_d, B_mu, B_V, B_d
                )

                if overlap_score > overlap_threshold:
                    edges.append((i, j, overlap_score))

                    if shared_entities:
                        if node_A.timestamp < node_B.timestamp:
                            node_A._newer_updates += 1
                            node_B._corroborators += 1
                        elif node_A.timestamp > node_B.timestamp:
                            node_B._newer_updates += 1
                            node_A._corroborators += 1
                        else:
                            node_A._corroborators += 1
                            node_B._corroborators += 1

        for node in retrieved_nodes:
            if node._newer_updates > 0:
                node.status = "OLDER STATE"
            elif node._corroborators > 0:
                node.status = "LATEST STATE"
            else:
                node.status = "VALID_ISOLATED"

        return edges


class GeometricRetriever:
    @staticmethod
    def get_top_k_geometric(query_node, candidate_nodes, k=10):
        if not candidate_nodes:
            return []

        q_mu, q_V, _, q_d = query_node.get_params()

        q_d_vec = q_d if q_d.dim() == 1 else torch.diagonal(q_d)
        q_sigma = torch.sqrt(q_d_vec + torch.sum(q_V**2, dim=-1))

        scored_candidates = []

        with torch.no_grad():
            for node in candidate_nodes:
                n_mu, n_V, _, n_d = node.get_params()

                n_d_vec = n_d if n_d.dim() == 1 else torch.diagonal(n_d)
                n_sigma = torch.sqrt(n_d_vec + torch.sum(n_V**2, dim=-1))

                mean_dist = torch.sum((q_mu - n_mu)**2)
                var_dist = torch.sum((q_sigma - n_sigma)**2)

                total_energy = mean_dist + var_dist

                scored_candidates.append((total_energy.item(), node))

        scored_candidates.sort(key=lambda x: x[0])

        top_k_nodes = [node for energy, node in scored_candidates[:k]]

        top_k_nodes.sort(key=lambda n: n.timestamp)

        return top_k_nodes
