import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class VGCFeatureExtractor(BaseFeaturesExtractor):
    # ability_dim is a kwarg that can be specified when creating the policy
    def __init__(self, observation_space, ability_dim=32, item_dim=32):
        # 1149 for the main features + 8 abilities each of size ability_dim + 8 items each of size item_dim
        features_dim = 1149 + (8 * ability_dim) + (8 * item_dim)
        super(VGCFeatureExtractor, self).__init__(observation_space, features_dim)
        # No bias to preserve 0 probability for impossible abilities
        self.ability_embedding = nn.Linear(215, ability_dim, bias=False)
        self.item_embedding = nn.Linear(167, item_dim, bias=False)

    def forward(self, observations):
        numeric_features = observations['numeric']
        abilities = observations['abilities']
        items = observations['items']
        # Encode abilities, items using the linear layer (8, 215) -> (8, 32) & (8, 167) -> (8, 32)
        embedded_abilities = self.ability_embedding(abilities)
        embedded_items = self.item_embedding(items)
        # Flatten the abilities to concatenate with numeric features (8, 32) -> (256)
        embedded_abilities_flat = embedded_abilities.flatten(start_dim=1)
        embedded_items_flat = embedded_items.flatten(start_dim=1)
        # Return the concatenated features (1149 + 256 + 256 = 1661)
        return torch.cat([numeric_features, embedded_abilities_flat, embedded_items_flat], dim=1)