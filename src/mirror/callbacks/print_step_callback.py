from mirror.callbacks.callback import Callback


class PrintStepCallback(Callback):
    def on_train_batch_end(self, fabric, model, optimizer, loss, tokens, attention_mask, training_run_id, batch_idx):
        print(f'iteration {batch_idx}, loss: {loss}')
