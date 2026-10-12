import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class VGCFeatureExtractor(BaseFeaturesExtractor):
    # ability_dim is a kwarg that can be specified when creating the policy
    def __init__(self, observation_space, ability_dim: int = 32, item_dim: int = 32):
        # 1149 for the main features + 8 abilities each of size ability_dim + 8 items each of size item_dim
        features_dim = 1149 + (8 * ability_dim) + (8 * item_dim)
        super().__init__(observation_space, features_dim)
        # No bias to preserve 0 probability for impossible abilities
        self.ability_embedding = nn.Linear(215, ability_dim, bias=False)
        self.item_embedding = nn.Linear(167, item_dim, bias=False)

    def forward(self, observations) -> torch.Tensor:
        numeric_features = observations['numeric']
        abilities = observations['abilities']
        items = observations['items']
        # Encode abilities, items using the linear layer (8, 215) -> (8, 32) & (8, 167) -> (8, 32)
        embedded_abilities = self.ability_embedding(abilities)
        embedded_items = self.item_embedding(items)
        # Flatten the abilities to concatenate with numeric features (8, 32) -> (8 * 32) = 256
        embedded_abilities_flat = embedded_abilities.flatten(start_dim=1)
        embedded_items_flat = embedded_items.flatten(start_dim=1)
        # Return the concatenated features (1149 + 256 + 256 = 1661)
        return torch.cat([numeric_features, embedded_abilities_flat, embedded_items_flat], dim=1)
    
class TeamPreviewModel(nn.Module):
    def __init__(self, ability_dim: int = 32, item_dim: int = 32):
        super().__init__()
        
        self.ability_embedding = nn.Linear(215, ability_dim, bias=False)
        self.item_embedding = nn.Linear(167, item_dim, bias=False)
        
        features_dim = 1704 + (12 * ability_dim) + (12 * item_dim) # 2472
        
        self.policy_head = nn.Sequential(
            nn.Linear(features_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 90),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, observations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        numeric = observations['numeric']
        abilities = observations['abilities']
        items = observations['items']

        is_unbatched = numeric.dim() == 1
        if is_unbatched:
            numeric = numeric.unsqueeze(0) # (1, 1704)
            abilities = abilities.unsqueeze(0) # (1, 12, 215)
            items = items.unsqueeze(0) # (1, 12, 167)

        embedded_abilities = self.ability_embedding(abilities).flatten(start_dim=1) # (batch, 384)
        embedded_items = self.item_embedding(items).flatten(start_dim=1) # (batch, 384)

        combined = torch.cat([numeric, embedded_abilities, embedded_items], dim=1) # (batch, 2472)
        policy_out = self.policy_head(combined) # (batch, 90)
        value_out = self.value_head(combined) # (batch, 1)

        if is_unbatched:
            return policy_out.squeeze(0), value_out.squeeze(0) # (90,), (1,)
        return policy_out, value_out
        