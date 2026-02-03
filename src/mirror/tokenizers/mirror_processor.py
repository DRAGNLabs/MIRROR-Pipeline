# from abc import ABC, abstractmethod
# from typing import Sequence, List

# import torch

# from mirror.types import TokenTensor, TokenBatch


# class MirrorProcessor(ABC):
#     @property
#     @abstractmethod
#     def _tokenizer(self):
#         pass

#     @abstractmethod
#     def preprocess_example(self, example):
#         pass

#     @abstractmethod
#     def collate(self, examples: list[ProcessedT]):
#         pass
    
