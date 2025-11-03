from callbacks.callback import Callback


class PrintStepCallback(Callback):
    def on_train_batch_end(self, fabric, model, loss, tokens, attention_mask, batch_idx):
        print(f'iteration {batch_idx}, loss: {loss.item()}')
