import torch.nn as nn


class DepthEmbedding(nn.Embedding):

    def __init__(self, embed_size=512):
        super(DepthEmbedding, self).__init__(6, embed_size, padding_idx=0)
