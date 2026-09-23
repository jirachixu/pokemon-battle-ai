import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class VGCFeatureExtractor(BaseFeaturesExtractor):
    # ability_dim is a kwarg that can be specified when creating the policy
    def __init__(self, observation_space, ability_dim=32):
        features_dim = 1149 + (8 * ability_dim)  # 1149 for the main features + 8 abilities each of size ability_dim
        super(VGCFeatureExtractor, self).__init__(observation_space, features_dim)
        # No bias to preserve 0 probability for impossible abilities
        self.ability_embedding = nn.Linear(215, ability_dim, bias=False) 

    def forward(self, observations):
        numeric_features = observations['numeric']
        abilities = observations['abilities']
        # Encode abilities using the linear layer (8, 215) -> (8, 32)
        embedded_abilities = self.ability_embedding(abilities)
        # Flatten the abilities to concatenate with numeric features (8, 32) -> (256)
        embedded_abilities_flat = embedded_abilities.flatten(start_dim=1)
        # Return the concatenated features (1149 + 256 = 1405)
        return torch.cat([numeric_features, embedded_abilities_flat], dim=1)