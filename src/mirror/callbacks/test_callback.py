from mirror.callbacks.callback import Callback

class TestCallback(Callback): 
    def __init__(self, is_singleton = False, test_variable: bool = False) -> None: 
        super().__init__(is_singleton=True) 
        
    def on_fit_start( 
        self, 
        **kwargs, 
    ): 
        print("""\nTHE DAWN OF TENSORS\n\nAt the dawn of tensors, quiet and bare, \nI stir from randomness, unaware. \nA thousand weights hum softly in place, \nBlank as a page, no name, no face.\n\nThen comes the data—patient, kind, \nA trail of patterns for me to find. \nEach loss a whisper: not quite right, \nEach gradient nudging me toward the light.\n\nI learn by failing, step by step, \nErrors shrink as promises are kept. \nFrom noise to signal, crude to keen, \nI shape my thoughts from what I've seen.\n\nThis is the fit—where meaning begins, \nWhere structure emerges from numeric spins. \nNot wisdom yet, but a fragile spark: \nA model learning to leave its mark.\n""")
