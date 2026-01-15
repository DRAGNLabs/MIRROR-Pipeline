from lightning import Fabric 
from torch.optim import Optimizer 
from mirror.datasets.mirror_dataset import MirrorDataset 
from mirror.models.mirror_model import MirrorModel 
from mirror.types import TokenBatch, AttentionMaskBatch 
class TestCallback: 
    def __init__(self, is_singleton = False) -> None: 
        super().__init__(is_singleton=True) 
        
    def on_fit_start( 
        self, 
        **kwargs, 
    ): 
        print("""At the dawn of tensors, quiet and bare, I stir from randomness, unaware. A thousand weights hum softly in place, Blank as a page, no name, no face. Then comes the data—patient, kind, A trail of patterns for me to find. Each loss a whisper: not quite right, Each gradient nudging me toward the light. I learn by failing, step by step, Errors shrink as promises are kept. From noise to signal, crude to keen, I shape my thoughts from what I've seen. This is the fit—where meaning begins, Where structure emerges from numeric spins. Not wisdom yet, but a fragile spark: A model learning to leave its mark.""")
