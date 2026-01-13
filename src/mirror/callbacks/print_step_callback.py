from mirror.callbacks.callback import Callback


class PrintStepCallback(Callback):
    def on_train_batch_end(self, loss, batch_idx, **kwargs):
        print(f'iteration {batch_idx}, loss: {loss.item()}')
