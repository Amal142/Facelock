"""
Module 5: Face Matching
Compares a live embedding against stored profiles using cosine similarity.
"""

import numpy as np
import logging
from modules.database import Database

logger = logging.getLogger(__name__)

# Cosine similarity threshold: values in [0, 1].
# InsightFace 512-D embeddings are L2-normalised, so dot-product == cosine sim.
# Typical sweet spot: 0.35 – 0.50  (raise to tighten, lower to loosen).
DEFAULT_THRESHOLD = 0.40


class Authenticator:
    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.db = Database()
        self.threshold = threshold

    # ------------------------------------------------------------------ #
    #  Enrolment                                                           #
    # ------------------------------------------------------------------ #

    def enroll(self, user_id: str, embeddings: list) -> bool:
        """
        Average a list of embeddings captured over several frames and save
        the result as the user's profile.

        Returns True on success.
        """
        if not embeddings:
            logger.warning("No embeddings provided for enrolment.")
            return False

        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)   # L2 normalise
        self.db.save_profile(user_id, mean_emb)
        return True

    # ------------------------------------------------------------------ #
    #  Authentication                                                      #
    # ------------------------------------------------------------------ #

    def identify(self, embedding: np.ndarray):
        """
        Compare embedding against all stored profiles.

        Returns:
            (user_id, similarity)  if the best match exceeds the threshold.
            (None,    best_score)  otherwise.
        """
        profiles = self.db.get_all()
        if not profiles:
            return None, 0.0

        # L2-normalise the query vector
        emb = embedding / np.linalg.norm(embedding)

        best_id, best_score = None, -1.0
        for uid, stored in profiles.items():
            score = float(np.dot(emb, stored))   # cosine similarity
            if score > best_score:
                best_score = score
                best_id = uid

        if best_score >= self.threshold:
            logger.info(f"Match: {best_id}  sim={best_score:.3f}")
            return best_id, best_score

        logger.debug(f"No match (best={best_score:.3f}, need >={self.threshold})")
        return None, best_score

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def list_users(self):
        return self.db.list_users()

    def delete_user(self, user_id: str):
        return self.db.delete(user_id)
