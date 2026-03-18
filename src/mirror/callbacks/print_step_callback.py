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

    def on_validation_epoch_end(
            self,
            *,
            val_loss,
            epoch,
            **kwargs
    ):
        print(f'epoch {epoch}, val_loss: {val_loss}')

    def on_test_epoch_end(
            self,
            *,
            test_loss,
            **kwargs
    ):
        print(f'test_loss: {test_loss}')
