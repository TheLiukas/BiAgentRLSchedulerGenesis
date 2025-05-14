import numpy as np
from collections import deque
import logging

# Optional: Set up logging to see when stopping occurs
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class EarlyStoppingCallback:
    """
    A callback to stop training if the mean reward doesn't improve significantly
    over a specified number of evaluation periods (epochs).

    :param patience: How many epochs to wait for improvement before stopping.
    :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
    :param warmup: Number of initial epochs to ignore before starting to monitor.
    :param verbose: If True, prints messages when checking and stopping.
    """
    def __init__(self,  patience: int = 10, min_delta: float = 0.1, warmup: int = 5, verbose: bool = True,):
        if patience <= 0:
            raise ValueError("Patience must be a positive integer.")
        if min_delta < 0:
             raise ValueError("min_delta must be non-negative.")
        if warmup < 0:
            raise ValueError("warmup must be non-negative.")

        
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.verbose = verbose

        # Use deque to automatically keep track of the last 'patience' rewards
        self.reward_history = deque(maxlen=self.patience)
        self.epoch_count = 0
        self.best_reward = -np.inf # Keep track of the best reward seen *after* warmup

    def setTrainer(self,trainer):
        self.trainer=trainer
    
    def __call__(self, mean_rewards: float) -> bool:
        """
        Call method executed by the Tianshou trainer.

        :param mean_rewards: The average reward obtained in the latest test phase.
        :return: True if training should stop, False otherwise.
        """
        if self.trainer==None:
            self.epoch_count += 1
        else :
            self.epoch_count=self.trainer.epoch
            #log.info(f"[EarlyStopping] Warmup epoch {self.epoch_count}/{self.warmup}. Current reward: {mean_rewards:.4f}")
        # --- Warmup Phase ---
        if self.epoch_count <= self.warmup:
            # During warmup, just record the reward and potentially update best_reward
            # We don't add to history yet as we don't check for stagnation
            self.best_reward = max(self.best_reward, mean_rewards)
            if self.verbose:
                log.info(f"[EarlyStopping] Warmup epoch {self.epoch_count}/{self.warmup}. Current reward: {mean_rewards:.4f}")
            return False # Don't stop during warmup

        # --- Monitoring Phase ---
        # Add current reward to history (deque handles the size limit)
        self.reward_history.append(mean_rewards)

        # Check if we have enough history to make a decision
        if len(self.reward_history) < self.patience:
            # Not enough history yet, update best_reward and continue
            self.best_reward = max(self.best_reward, mean_rewards)
            if self.verbose:
                 log.info(f"[EarlyStopping] Collecting history ({len(self.reward_history)}/{self.patience}). Current reward: {mean_rewards:.4f}")
            return False

        # --- Stagnation Check ---
        # Check if the reward 'patience' epochs ago plus delta is greater than all rewards since then
        # An alternative: Check if the *best* reward seen recently is not much better than oldest in window
        # Simpler check: Is the current reward significantly better than the oldest reward in the window?
        oldest_reward = self.reward_history[0] # Reward from 'patience' epochs ago

        # More robust check: Has the *best* reward seen in the last 'patience' epochs improved
        # significantly compared to the best reward seen *before* this window started?
        # Let's stick to a simpler check first: Lack of improvement over the window.
        # Improvement = current_reward - oldest_reward
        # If improvement < min_delta, we might stop.

        # Refined check: Consider the best reward in the current window
        best_reward_in_window = max(self.reward_history)

        # Check 1: Has the best reward overall improved recently?
        if best_reward_in_window > self.best_reward + self.min_delta:
             # Significant improvement found, update best reward and reset patience implicitly
             if self.verbose:
                  log.info(f"[EarlyStopping] Improvement detected! Best reward updated from {self.best_reward:.4f} to {best_reward_in_window:.4f}")
             self.best_reward = best_reward_in_window
             # Even though deque is fixed size, finding improvement means we shouldn't stop
             return False # Continue training

        # Check 2: If no significant improvement overall, check if we've waited long enough
        # At this point, we know:
        # 1. We are past the warmup phase.
        # 2. We have 'patience' entries in our history.
        # 3. The best reward in the last 'patience' epochs is NOT significantly better
        #    than the best reward seen before this window. -> Implies stagnation.
        if self.verbose:
            log.warning(f"[EarlyStopping] No significant improvement for {self.patience} epochs (current: {mean_rewards:.4f}, best_in_window: {best_reward_in_window:.4f}, overall_best: {self.best_reward:.4f}). Stopping.")
        return True # Stop training