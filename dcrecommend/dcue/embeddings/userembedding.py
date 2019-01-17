"""PyTorch class for user embedding in DCUE model."""

import torch.nn as nn


class UserEmbeddings(nn.Module):

    """Class to embed users as feature vector."""

    def __init__(self, dict_args):
        """
        Initialize UserEmbeddings.

        Args
            dict_args: dictionary containing the following keys:
                user_embdim: The dimension of the lookup embedding.
                user_count: The count of users in the data set.
                feature_dim: The dimension of the feature vector to embed
                    the users in.
        """
        super(UserEmbeddings, self).__init__()

        self.user_embdim = dict_args["user_embdim"]
        self.user_count = dict_args["user_count"]
        self.feature_dim = dict_args["feature_dim"]

        self.embeddings = nn.Embedding(self.user_count, self.user_embdim)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(self.user_embdim, self.user_embdim)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(self.user_embdim, self.feature_dim)

    def forward(self, user_idx):
        """
        Forward pass.

        Args
            user_idx: A tensor of user indexes of size batch_size x 1.
        """
        user = self.embeddings(user_idx)  # batch_size x 1 x user_embdim
        user = self.relu1(user)
        user = self.linear1(user)
        user = self.relu2(user)
        return self.linear2(user)


# if __name__=='__main__':
#
# 	dict_args = {
# 					"user_embdim" : 300,
# 					"user_count": 10,
#                     "feature_dim": 100
# 				}

    # user_embeddings = UserEmbeddings(dict_args)
    # print(user_embeddings(torch.LongTensor([[1],[9]])))
    # print(user_embeddings(torch.LongTensor([[1],[9]])).size())
