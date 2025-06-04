import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np


class MDescAug(nn.Module):
    """
    Top-M Descriptor Augmentation (MDA) según el paper:

    Shao, S., Chen, K., Karpur, A., Cui, Q., Araujo, A., Cao, B.:
    "Global features are all you need for image retrieval and reranking". ICCV (2023)

    Este módulo toma los descriptores de las imágenes top-M para cada query,
    calcula relaciones internas entre ellos y genera una nueva versión refinada
    de los descriptores mediante una agregación ponderada.

    Args:
        M (int): Número de vecinos iniciales top-M a considerar.
        K (int): Número de vecinos internos a usar por cada top-M (incluyéndose a sí mismo).
        beta (float): Peso que se asigna a los vecinos (excepto uno mismo).
    """
    def __init__(self, M=400, K=9, beta=0.15):
        super(MDescAug, self).__init__()
        self.M = M
        self.K = K + 1  # incluyendo a uno mismo
        self.beta = beta

    def forward(self, X, Q, ranks):
        """
        Ejecuta el refinamiento de descriptores basado en top-M y similitud interna.

        Args:
            X (Tensor): Embeddings de la base, forma (N, d)
            Q (Tensor): Embeddings de las queries, forma (q, d)
            ranks (Tensor): Ranking original, forma (N, q)

        Returns:
            tuple:
                - rerank_dba_final (list[Tensor]): Nuevos índices ordenados top-M para cada query
                - res_top1000_dba (Tensor): Similitudes Q · x_dba (q, M)
                - ranks_trans_1000_pre (Tensor): Índices locales ordenados según res_top1000_dba (q, M)
                - x_dba (Tensor): Embeddings refinados para top-M (q, M, d)
        """
        # Extraer top-M resultados por query
        ranks_trans_1000 = torch.transpose(ranks, 1, 0)[:, :self.M]  # (q, M)

        # Obtener embeddings X para los top-M de cada query
        X_tensor1 = X[ranks_trans_1000]  # (q, M, d)

        # Calcular similitud interna entre los top-M (por query)
        res_ie = torch.einsum('abc,adc->abd', X_tensor1, X_tensor1)  # (q, M, M)

        # Obtener los K vecinos más similares para cada uno de los M (incluido uno mismo)
        res_ie_ranks = torch.argsort(-res_ie.clone(), dim=-1)[:, :, :self.K]  # (q, M, K)
        res_ie_ranks_value = torch.sort(res_ie.clone(), dim=-1, descending=True)[0][:, :, :self.K].unsqueeze(-1)  # (q, M, K, 1)

        # Asignar peso 1 al propio embedding y beta al resto
        res_ie_ranks_value[:, :, 1:, :] *= self.beta
        res_ie_ranks_value[:, :, 0:1, :] = 1.

        # Recuperar embeddings vecinos para cada top-M
        x_dba_list = []
        for i, j in zip(res_ie_ranks, X_tensor1):
            x_dba_list.append(j[i])  # (M, K, d)

        x_dba = torch.stack(x_dba_list, dim=0)  # (q, M, K, d)

        # Agregación ponderada
        x_dba = torch.sum(x_dba * res_ie_ranks_value, dim=2) / torch.sum(res_ie_ranks_value, dim=2)  # (q, M, d)

        # Calcular similitud con embeddings refinados
        res_top1000_dba = torch.einsum('ac,adc->ad', Q, x_dba)  # (q, M)

        # Ordenar nuevamente los top-M según la nueva similitud
        ranks_trans_1000_pre = torch.argsort(-res_top1000_dba, dim=-1)  # (q, M)

        # Convertir índices locales a índices reales del ranking original
        rerank_dba_final = []
        for i in range(ranks_trans_1000_pre.shape[0]):
            temp_concat = ranks_trans_1000[i][ranks_trans_1000_pre[i]]
            rerank_dba_final.append(temp_concat)  # (M,)

        return rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba

class RerankwMDA(nn.Module):
    """
    Reranking with Maximum Descriptor Aggregation (MDA) según el paper:

    Shao, S., Chen, K., Karpur, A., Cui, Q., Araujo, A., Cao, B.:
    "Global features are all you need for image retrieval and reranking". ICCV (2023)

    Esta clase toma como entrada:
      - el ranking top-M refinado para cada query,
      - las similitudes entre la query y esos nuevos descriptores,
      - y los embeddings refinados de la base,
    para producir un nuevo ranking final extendido a toda la base.
    """
    def __init__(self, M=400, K=9, beta=0.15):
        super(RerankwMDA, self).__init__()
        self.M = M
        self.K = K + 1  # incluyendo a uno mismo
        self.beta = beta

    def forward(self, ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba):
        """
        Reordena los rankings globales con base en la similitud entre cada query
        y un embedding agregado (máximo) construido a partir de los top-K vecinos.

        Args:
            ranks (Tensor): Ranking original de la base, (N, q)
            rerank_dba_final (list[Tensor]): Índices de top-M rerankeados, uno por query, (q, M)
            res_top1000_dba (Tensor): Similitudes Q · x_dba, (q, M)
            ranks_trans_1000_pre (Tensor): Ranking interno (local) previo dentro de los M, (q, M)
            x_dba (Tensor): Embeddings refinados de top-M, (q, M, d)

        Returns:
            Tensor: Ranking final extendido para cada query, (N, q)
        """
        # Índices reales de los M elementos rerankeados
        ranks_trans_1000 = torch.stack(rerank_dba_final, dim=0)  # (q, M)

        # Similitud original con esos M elementos
        ranks_value_trans_1000 = -torch.sort(-res_top1000_dba, dim=-1)[0]  # (q, M)

        # Extraer los K vecinos más similares dentro de top-M, y ponderarlos con beta
        ranks_trans = ranks_trans_1000_pre[:, :self.K].unsqueeze(-1)  # (q, K, 1)
        ranks_value_trans = ranks_value_trans_1000[:, :self.K].clone().unsqueeze(-1)  # (q, K, 1)
        ranks_value_trans *= self.beta  # todos los vecinos tienen peso beta

        # Obtener descriptores refinados X1 (máximo entre K vecinos) y X2 (todo top-M)
        X1 = torch.take_along_dim(x_dba, ranks_trans.expand(-1, -1, x_dba.size(-1)), dim=1)  # (q, K, d)
        X2 = torch.take_along_dim(x_dba, ranks_trans_1000_pre.unsqueeze(-1).expand(-1, -1, x_dba.size(-1)), dim=1)  # (q, M, d)
        X1 = torch.max(X1, dim=1, keepdim=True)[0]  # (q, 1, d)

        # Similitud entre el max descriptor y cada uno del top-M
        res_rerank = torch.sum(torch.einsum('abc,adc->abd', X1, X2), dim=1)  # (q, M)

        # Fusión con la similitud original
        res_rerank = (ranks_value_trans_1000 + res_rerank) / 2.0  # (q, M)
        res_rerank_ranks = torch.argsort(-res_rerank, dim=-1)  # (q, M)

        # Concatenar el top-M rerankeado con el resto del ranking original
        rerank_qe_final = []
        ranks_transpose = torch.transpose(ranks, 1, 0)[:, self.M:]  # (q, N-M)
        for i in range(res_rerank_ranks.shape[0]):
            temp_concat = torch.cat([
                ranks_trans_1000[i][res_rerank_ranks[i]],  # nuevo top-M
                ranks_transpose[i]                          # resto original
            ], dim=0)  # (N,)
            rerank_qe_final.append(temp_concat)

        # Convertir a forma estándar: (N, q)
        ranks_final = torch.transpose(torch.stack(rerank_qe_final, dim=0), 1, 0)
        return ranks_final

class SGReranker:
    """
    Refinador de resultados que implementa el reranking con SuperGlobal Descriptor Augmentation (MDA),
    según el paper:

    Shao, S., Chen, K., Karpur, A., Cui, Q., Araujo, A., Cao, B.:
    "Global features are all you need for image retrieval and reranking". ICCV (2023)

    Este método aplica dos pasos:
      1. MDescAug: fusión de descriptores de los top-M vecinos con pesos adaptativos.
      2. RerankwMDA: reordenamiento basado en similitud con un descriptor máximo fusionado.

    Se utiliza tras la búsqueda global inicial para refinar los rankings.
    """
    def __init__(self, M=400, K=9, beta=0.15):
        self.M = M
        self.K = K
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mda = MDescAug(M=M, K=K, beta=beta).to(self.device)
        self.reranker = RerankwMDA(M=M, K=K, beta=beta).to(self.device)

    def refine(self, X_np, Q_np, ranks_np):
        """
        Refina el ranking usando reranking con MDescAug y RerankwMDA.

        Args:
            X_np (np.ndarray): Embeddings base, (N, d)
            Q_np (np.ndarray): Embeddings de consulta, (q, d)
            ranks_np (np.ndarray): Ranking original, (N, q)

        Returns:
            np.ndarray: Nuevo ranking (N, q)
        """
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        Q = torch.tensor(Q_np, dtype=torch.float32, device=self.device)
        ranks = torch.tensor(ranks_np, dtype=torch.int64, device=self.device)

        rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = self.mda(X, Q, ranks)
        new_ranks = self.reranker(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)

        return new_ranks.cpu().numpy()

    def refine(self, X_np, Q_np, ranks_np):
        """
        Refina el ranking usando reranking con MDescAug y RerankwMDA.

        Como en el pipeline original las queries se excluyen de X, aquí se corrige esa situación
        concatenando Q al inicio de X, y ajustando los índices del ranking para asegurar que el
        propio descriptor de la query esté incluido como top-1.

        Args:
            X_np (np.ndarray): Embeddings base, de forma (N, d)
            Q_np (np.ndarray): Embeddings de consulta, de forma (q, d)
            ranks_np (np.ndarray): Ranking original, de forma (N, q)

        Returns:
            np.ndarray: Nuevo ranking, de forma (N, q)
        """
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        Q = torch.tensor(Q_np, dtype=torch.float32, device=self.device)
        ranks = torch.tensor(ranks_np, dtype=torch.int64, device=self.device)

        q = Q.shape[0]  # número de queries
        N = X.shape[0]  # tamaño base

        # Concatenar Q al inicio de X para formar la base extendida
        X_full = torch.cat([Q, X], dim=0)  # (q + N, d)

        # Desplazar todos los índices del ranking original en +q
        ranks_shifted = ranks + q  # (N, q)

        # Construir nuevo ranking incluyendo a cada query como primer resultado
        q_indices = torch.arange(q, device=self.device).unsqueeze(0)  # (1, q)
        ranks_adjusted = torch.cat([q_indices, ranks_shifted], dim=0)  # (N+1, q)

        # Aplicar MDA y reranking
        rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = self.mda(X_full, Q, ranks_adjusted)
        new_ranks_extended = self.reranker(ranks_adjusted, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)

        # Corregir los índices finales restando q para volver a la referencia original
        new_ranks = new_ranks_extended - q

        # Eliminar cualquier índice negativo (que representa las queries originales)
        mask = new_ranks >= 0
        new_ranks = torch.masked_select(new_ranks, mask).reshape(N, q)

        return new_ranks.cpu().numpy()
