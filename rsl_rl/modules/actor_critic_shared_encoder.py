# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any
import warnings

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization ,CNN_1D,HiddenState, Memory

from .actor_critic import ActorCritic


class ActorCriticShared(nn.Module):
    is_recurrent: bool = True
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        shared_obs_normalization: bool = False,
        stohastic_encoder :  bool = False,
        shared_network_dims : tuple[int] | list[int] = [256, 128],
        shared_hidden_dims: tuple[int] | list[int] = [128],
        actor_hidden_dims: tuple[int] | list[int] = [128],
        critic_hidden_dims: tuple[int] | list[int] = [128],
        shared_cnn_cfg: dict[str, dict] | dict | None = None,
        code_size : int = 128,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        shared_endcoder: bool = True,
        safety_critic: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
    
        super().__init__()

        if shared_endcoder== False:
            raise ValueError(f"Method we seperate encoder not implemented yet shared_endcoder: {shared_endcoder}")
        # Get the observation dimensions
        print(obs_groups," that is obs shape: ", {k: obs[k].shape for k in obs.keys()})
        self.obs_groups = obs_groups
        num_shared_obs_1d = 0
        self.shared_obs_groups_1d = []
        shared_in_dims_cnn = []
        shared_in_channels_cnn = []
        self.shared_obs_groups_cnn = []
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) == 3:  # B, C, W
                self.shared_obs_groups_cnn.append(obs_group)
                shared_in_dims_cnn.append(obs[obs_group].shape[2])
                shared_in_channels_cnn.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.shared_obs_groups_1d.append(obs_group)
                num_shared_obs_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")
    

        # Actor CNN
        if self.shared_obs_groups_cnn:
            # Resolve the actor CNN configuration
            assert shared_cnn_cfg is not None, "An actor CNN configuration is required for 2D actor observations."
            # If a single configuration dictionary is provided, create a dictionary for each 2D observation group
            print("shared_cnn_cfg before:", shared_cnn_cfg)
            if not all(isinstance(v, dict) for v in shared_cnn_cfg.values()):
                shared_cnn_cfg = {group: shared_cnn_cfg for group in self.shared_obs_groups_cnn}
            # Check that the number of configs matches the number of observation groups
            assert len(shared_cnn_cfg) == len(self.shared_obs_groups_cnn), (
                "The number of CNN configurations must match the number of 2D actor observations."
            )

            # Create CNNs for each cnn actor observation
            self.shared_cnns = nn.ModuleDict()
            encoding_dim = 0
            for idx, obs_group in enumerate(self.shared_obs_groups_cnn):
                self.shared_cnns[obs_group] = CNN_1D(
                    input_dim=shared_in_dims_cnn[idx],
                    input_channels=shared_in_channels_cnn[idx],
                    **shared_cnn_cfg[obs_group],
                )
                print(f"Shared CNN for {obs_group}: {self.shared_cnns[obs_group]}")
                # Get the output dimension of the CNN
                if self.shared_cnns[obs_group].output_channels is None:
                    encoding_dim += int(self.shared_cnns[obs_group].output_dim)
                else:
                    raise ValueError("The output of the actor CNN must be flattened before passing it to the MLP.")
        else:
            self.shared_cnns = None
            encoding_dim = 0

        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        

        # Actor
        self.state_dependent_std = state_dependent_std
        self.memory_shared = Memory(num_shared_obs_1d + encoding_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        print(f"rnn_hidden_dim:{rnn_hidden_dim}, code_size:{code_size}, shared_hidden_dims:{shared_hidden_dims}")
        if stohastic_encoder:
            shared_output = code_size*2  #2*code size because we get mean and variance output for the features.
        else:
            shared_output = code_size
        self.shared_MLP = MLP(rnn_hidden_dim, shared_output, shared_hidden_dims, activation) 
        
        print(f"Shared RNN: {self.memory_shared}")
        print(f"Shared MLP: {self.shared_MLP}")

        # Shared observation normalization
        self.shared_obs_normalization = shared_obs_normalization
        if shared_obs_normalization:
            self.shared_obs_normalizer = EmpiricalNormalization(num_shared_obs_1d)
        else:
            self.shared_obs_normalizer = torch.nn.Identity()     

        if self.state_dependent_std:
            self.actor = MLP(shared_output, [2, num_actions],actor_hidden_dims, activation)
        else:
            self.actor = MLP(shared_output, num_actions, actor_hidden_dims, activation)

   

        self.critic = MLP(shared_output, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        if safety_critic:
            self.safety_critic = MLP(shared_output, 1, critic_hidden_dims, activation)
            print(f"Safety Critic MLP: {self.safety_critic}")



        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)



    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_shared.reset(dones)
        
       
    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()
    
    def extract_features(self,obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None)->torch.Tensor:
        mlp_obs, cnn_obs = self.get_shared_obs(obs)
        if self.shared_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = []
            for obs_group in self.shared_obs_groups_cnn:
                x = cnn_obs[obs_group]

                # If sequence dimension exists → merge seq and batch
                if x.dim() == 4:        # (seq, batch, C, L)
                    seq, b, C, L = x.shape
                    if b==0:
                        raise ValueError("Batch size is zero")
                    x = x.reshape(seq * b, C, L)
                    enc = self.shared_cnns[obs_group](x)
                    enc = enc.reshape(seq, b, -1)

                elif x.dim()==3:                   # (batch, C, L)
                    enc = self.shared_cnns[obs_group](x)
                    enc = enc.unsqueeze(0)   # → (1, batch, feat)
                else:
                    raise ValueError("not implemented yet")
                cnn_enc_list.append(enc)
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            if mlp_obs.dim() != cnn_enc.dim():
                cnn_enc=cnn_enc.squeeze()
            #cnn_enc_list = [self.shared_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.shared_obs_groups_cnn]
            #cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # Concatenate to the MLP observations
        mlp_obs = self.shared_obs_normalizer(mlp_obs)
        combined_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)
        out_mem = self.memory_shared(combined_obs, masks, hidden_state).squeeze(0)
        code = self.shared_MLP(out_mem)
        return code
    

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_shared_obs(obs)
        if self.shared_cnns is not None:
            # Encode the 2D actor observations
            cnn_enc_list = [self.shared_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.shared_obs_groups_cnn]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
        mlp_obs = self.shared_obs_normalizer(mlp_obs)
        combined_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)
        assert combined_obs is not None
        out_mem = self.memory_shared(combined_obs).squeeze(0)
        out_mlp = self.shared_MLP(out_mem)
        if self.state_dependent_std:
            return self.actor(out_mlp)[..., 0, :]
        else:
            return self.actor(out_mlp)

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)
    
    def safety_evaluate(self, obs: torch.Tensor) -> torch.Tensor: #used for constrained RL where a safety critic is needed.
        return self.safety_critic(obs)

    def get_shared_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list_1d = [obs[obs_group] for obs_group in self.shared_obs_groups_1d]
        obs_dict_2d = {}
        for obs_group in self.shared_obs_groups_cnn:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return self.memory_shared.hidden_state,self.memory_shared.hidden_state

    def update_normalization(self, obs: TensorDict) -> None:
        if self.shared_obs_normalization:
            shared_obs = self.get_shared_obs(obs)
            self.shared_obs_normalizer.update(shared_obs) 
     
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
