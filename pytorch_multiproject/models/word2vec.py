import torch
import torch.nn as nn


class Word2VecModel(nn.Module):

    def __init__(self, vocab_size, padding_idx=0, embedding_size=300, n_negatives=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_negatives = n_negatives
        self.input_embeddings = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.target_embeddings = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)

        self.input_embeddings.weight = nn.Parameter(
            torch.cat(
                [torch.zeros(1, self.embedding_size),
                 torch.FloatTensor(self.vocab_size - 1, self.embedding_size).uniform_(-0.5/self.embedding_size, 0.5/self.embedding_size)]
            ),
            requires_grad=True)

        self.target_embeddings.weight = nn.Parameter(
            torch.cat(
                [torch.zeros(1, self.embedding_size),
                 torch.FloatTensor(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]
            ),
            requires_grad=True)

    def forward(self, input_, target):
        device = input_.device
        batch_size = input_.shape[0]
        context_size = target.shape[1]
        # generate examples of negative context using uniform distribution
        negative_words = (torch.FloatTensor(batch_size, context_size * self.n_negatives)
                          .uniform_(0, self.vocab_size-1)
                          .long()
                          .to(device))
        inpt_emb = self.input_embeddings(input_)
        target_emb = self.target_embeddings(target)
        neg_emb = self.target_embeddings(negative_words).neg()

        true_loss = torch.bmm(target_emb, inpt_emb).squeeze().sigmoid().log().mean(1)
        negative_loss = (torch.bmm(neg_emb, inpt_emb).squeeze().sigmoid().log()
                         .view(-1, context_size, self.n_negatives)
                         .sum(2)
                         .mean(1))
        loss = - (true_loss + negative_loss).mean()
        return loss
