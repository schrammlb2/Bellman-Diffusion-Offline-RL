import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class SequentialBuffer:
    def __init__(
        self,
        buffer_size: int,
        samples: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self._samples = samples
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.future_obs = np.zeros((self._max_size, self._samples) + self.obs_shape, dtype=obs_dtype)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        import ipdb
        ipdb.set_trace()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        import ipdb
        ipdb.set_trace()
        # self.next_actions[indexes] = np.array(
        #     np.cat([actions[1:], actions[:-1]])
        # ).copy()
        # self.valid_next_actions[indexes] = np.array(valid_next_actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        next_actions = np.array(dataset["next_actions"], dtype=self.action_dtype)
        valid_next_actions = np.array(dataset["valid_next_actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        future_obs = np.array(dataset["future_obs"], dtype=self.obs_dtype)
        # valid_future_obs = np.expand_dims(np.array(dataset["valid_future_obs"], dtype=np.float32), axis=-1)
        valid_future_obs = np.array(dataset["valid_future_obs"], dtype=np.bool_).reshape(-1, self._samples, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.next_actions = next_actions
        self.valid_next_actions = valid_next_actions
        self.rewards = rewards
        self.terminals = terminals

        self.future_obs = future_obs
        self.valid_future_obs = valid_future_obs

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        self.future_obs = (self.future_obs - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "next_actions": torch.tensor(self.next_actions[batch_indexes]).to(self.device),
            "valid_next_actions": torch.tensor(self.valid_next_actions[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device), 

            "future_obs": torch.tensor(self.future_obs[batch_indexes]).to(self.device),
            "valid_future_obs": torch.tensor(self.valid_future_obs[batch_indexes]).to(self.device),

        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "next_actions": self.next_actions[:self._size].copy(),
            "valid_next_actions": self.valid_next_actions[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy(),

            "future_obs": self.future_obs[:self._size].copy(),
            "valid_future_obs": self.valid_future_obs[:self._size].copy(),
        }

class RuntimeSequentialBuffer:
    def __init__(
        self,
        buffer_size: int,
        samples: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        gamma: float,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self._samples = samples
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.gamma = gamma

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        import ipdb
        ipdb.set_trace()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        import ipdb
        ipdb.set_trace()
        # self.next_actions[indexes] = np.array(
        #     np.cat([actions[1:], actions[:-1]])
        # ).copy()
        # self.valid_next_actions[indexes] = np.array(valid_next_actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        next_actions = np.array(dataset["next_actions"], dtype=self.action_dtype)
        valid_next_actions = np.array(dataset["valid_next_actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.next_actions = next_actions
        self.valid_next_actions = valid_next_actions
        self.rewards = rewards
        self.terminals = terminals

        self.cum_terminals = np.cumsum(terminals, axis=0)


        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)

        future_obs_list = []
        valid_future_obs_list = []
        # for i in range(self._samples):
        #     offset = np.random.geometric(1-self.gamma, size=batch_size) + 1
        #     future_ind = batch_indexes + offset
        #     # valid_list = []
        #     #If this is a bottleneck, rewrite as scatter or gather operation
        #     # for index in range(batch_size):
        #     #     slice= self.terminals[batch_indexes[index]:future_ind[index]]
        #     #     valid = (future_ind[index] < terminals.shape[0]) and (slice.sum() == 0)
        #     #     valid_list.append(valid)
        #     # future_ind = np.clip(future_ind, a_min=0, a_max=terminals.shape[0]-1)
        #     # future_obs = observations[future_ind]
        #     # future_obs_list.append(future_obs)
        #     # valid_future_obs_list.append(np.array(valid_list))

        #     # clipped_future_ind = np.min(future_ind, np.ones(batch_size)*(self._size - 1))
        #     clipped_future_ind = np.clip(future_ind, a_min=0, a_max=self._size-1)
        #     valid = (
        #         (future_ind < self._size).reshape(-1, 1)#Not past end of data set
        #         * #And
        #         (self.cum_terminals[clipped_future_ind] - self.cum_terminals[batch_indexes] == 0)
        #         #No terminals between current and future index
        #     )
        #     future_obs = self.observations[clipped_future_ind]
        #     future_obs_list.append(future_obs)
        #     valid_future_obs_list.append(valid)

        offset = np.random.geometric(1-self.gamma, size=(batch_size,self._samples)) + 1
        future_ind = batch_indexes.reshape(-1,1) + offset
        clipped_future_ind = np.clip(future_ind, a_min=0, a_max=self._size-1)
        stacked_batch_indexes = np.stack([batch_indexes]*self._samples, axis=1)
        # tiled_batch_indexes = np.tile(batch_indexes, axis=1)
        valid = (
            (future_ind < self._size).reshape(batch_size, self._samples, 1)#Not past end of data set
            * #And
            # (self.cum_terminals[clipped_future_ind] - self.cum_terminals[batch_indexes] == 0)
            (self.cum_terminals[clipped_future_ind] - self.cum_terminals[stacked_batch_indexes] == 0)
            #No terminals between current and future index
        )
        future_obs_array = self.observations[clipped_future_ind]
        valid_future_obs_array = valid
        # future_obs_list.append(future_obs)
        # valid_future_obs_list.append(valid)

        
        # future_obs_array = np.stack(future_obs_list, axis=1)
        # valid_future_obs_array = np.stack(valid_future_obs_list, axis=1)
        # print(valid_future_obs_array.mean())
        # import ipdb
        # ipdb.set_trace()
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "next_actions": torch.tensor(self.next_actions[batch_indexes]).to(self.device),
            "valid_next_actions": torch.tensor(self.valid_next_actions[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device), 

            # "future_obs": torch.tensor(self.future_obs[batch_indexes]).to(self.device),
            # "valid_future_obs": torch.tensor(self.valid_future_obs[batch_indexes]).to(self.device),
            "future_obs": torch.tensor(future_obs_array).to(self.device),
            "valid_future_obs": torch.tensor(valid_future_obs_array).to(self.device),

        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "next_actions": self.next_actions[:self._size].copy(),
            "valid_next_actions": self.valid_next_actions[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy(),

            "future_obs": self.future_obs[:self._size].copy(),
            "valid_future_obs": self.valid_future_obs[:self._size].copy(),
        }