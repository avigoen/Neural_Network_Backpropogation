from constants import LR_RATE, EPSILON


class ReduceLRCallback:
    def __init__(self, lr=LR_RATE, patience=10, delta=EPSILON):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.lr = lr
        self.best_loss = None
        self.reduce_lr = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.reduce_lr = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def get_lr(self):
        return self.lr

    def reset(self):
        self.lr /= 10
        self.reduce_lr = False
