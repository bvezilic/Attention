class History:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_scores = []
        self.test_scores = []

    def update(self,
               train_loss: float,
               train_score: float,
               test_loss: float = None,
               test_score: float = None):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

        if test_score is None:
            self.train_scores.append(train_score)
        if test_loss is None:
            self.test_scores.append(test_score)
