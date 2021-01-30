import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Callable


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, transform_fn: Callable[[str], List[int]]) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            transform_fn (Callable): a function that can act as a transformer
        """
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._transform = transform_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int):
        # 토큰 인덱스 저장
        t2iL = []
        t2i = self._transform(self._corpus.iloc[idx]['document'])
        t2iL.append(t2i)
        return t2iL
