from mirror.callbacks.callback import Callback


class PrintStepCallback[RawT, ProcessedT, BatchT, ModelOutputT](
    Callback[RawT, ProcessedT, BatchT, ModelOutputT]
):
    def on_train_batch_end(
            self,
            *, 
            loss, 
            batch_idx,
            **kwargs
    ):
        print(f'iteration {batch_idx}, loss: {loss}')
