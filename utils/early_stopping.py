import copy


class EarlyStopping:
    """
    Validation-based early stopping utility.
    Tracks the best validation metric and stops search when
    it has not improved for 'patience' iterations.

    Used by hyperparameter.py's staged_search_strategy() to halt
    the search loop when validation Sharpe stops improving.

    Note: internally tracks 'best_val_loss' for legacy compatibility,
    but when used with mode='max_sharpe' the caller passes -sharpe
    so that minimising loss is equivalent to maximising Sharpe.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience         = patience
        self.min_delta        = min_delta
        self.best_val_loss    = float("inf")
        self.best_epoch       = None
        self.best_model_state = None   # stores best StrategySession state
        self.counter          = 0
        self.stopped_epoch    = None
        self.triggered        = False

    def update(self, val_loss: float, model, epoch: int) -> bool:
        """
        Update early stopping state for a new iteration.
        Returns True if the search/training should stop.

        Args:
            val_loss : Current validation metric (lower is better).
                       For Sharpe maximisation pass -sharpe_value.
            model    : Current session/model object. State is deepcopied on improvement.
            epoch    : Current iteration number (epoch or bar_idx).

        Returns:
            bool: True if patience exceeded and should stop.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss    = val_loss
            self.best_epoch       = epoch
            self.best_model_state = copy.deepcopy(model)
            self.counter          = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            self.triggered     = True
            return True

        return False
