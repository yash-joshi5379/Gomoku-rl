# src/logger.py
import pandas as pd
from pathlib import Path
from src.config import Config


class Logger:
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir or Config.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data = []

    def log_episode(self, stats):
        """Stats is a dict with: episode, outcome, reward, loss, epsilon, buffer"""
        self.data.append(stats)

    def save(self):
        """Save everything to training.csv in correct importance order"""
        if self.data:
            df = pd.DataFrame(self.data)
            cols = ["episode", "outcome", "reward", "loss", "epsilon", "buffer"]
            existing = [c for c in cols if c in df.columns]
            df[existing].to_csv(self.log_dir / "training.csv", index=False)
