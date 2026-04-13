"""
Module 4: Profile Storage
Saves and loads face embeddings to/from a simple pickle file.
No encryption — focus is purely on facial recognition.
"""

import os
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEFAULT_PATH = "data/profiles.pkl"


class Database:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._profiles = self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def save_profile(self, user_id: str, embedding: np.ndarray):
        """Store (or overwrite) a user's embedding."""
        self._profiles[user_id] = embedding
        self._persist()
        logger.info(f"Profile saved: {user_id}")

    def get_all(self) -> dict:
        """Return {user_id: embedding} for every enrolled user."""
        return dict(self._profiles)

    def delete(self, user_id: str) -> bool:
        """Remove a user. Returns True if the user existed."""
        if user_id in self._profiles:
            del self._profiles[user_id]
            self._persist()
            logger.info(f"Profile deleted: {user_id}")
            return True
        return False

    def list_users(self) -> list:
        return list(self._profiles.keys())

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded {len(data)} profile(s) from {self.path}")
            return data
        return {}

    def _persist(self):
        with open(self.path, "wb") as f:
            pickle.dump(self._profiles, f)
