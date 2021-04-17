
import numpy as np

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance)


class PerspRegBased(VisionPruning):
    def __init__(self, model,experiment, inputs=None, outputs=None, compression=1):

        super().__init__(model, inputs, outputs, compression=compression)
        self.experiment=experiment
        

    def model_masks(self):
        importances = map_importances(np.abs, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        breakpoint()
        return masks