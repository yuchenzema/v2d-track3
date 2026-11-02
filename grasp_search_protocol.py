"""Reproducible independent randomness for each named grasp-search attempt."""
import hashlib


def search_seed(episode, label):
    return int.from_bytes(hashlib.sha256(f'{int(episode)}:{label}'.encode()).digest()[:4], 'little')
