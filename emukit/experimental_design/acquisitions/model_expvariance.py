# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

import numpy as np

from ...core.acquisition import Acquisition
from ...core.interfaces import IModel, IDifferentiable


class ModelExpVariance(Acquisition):
    """
    This acquisition selects the point in the domain where the predictive variance is the highest
    """
    def __init__(self, model: Union[IModel, IDifferentiable]) -> None:
        self.model = model

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        mean, variance = self.model.predict(x)
        return np.sqrt((np.exp(mean[0])*variance)**2) # true in uncorrelated limit
        #return np.exp(2*mean + variance**2)*(np.exp(variance**2)-1) #Kandasamy utility function
        #return np.exp(mean)*np.exp(2*mean + variance**2)*(np.exp(variance**2)-1) #Kandasamy utility function

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        mean, variance = self.model.predict(x)
        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)

        #exp_dvariance_dx = mean*np.exp(mean)*dvariance_dx #naive chain rule
        #expvar = np.sqrt((np.exp(mean)*variance)**2)
        #"""** THIS IS ALL NOW JUST MEAN REVERTING AND IS HORRIBLE**"""

        #return , exp_dvariance_dx
        #return np.exp(2*mean + variance**2)*(np.exp(variance**2)-1), dvariance_dx
        #return np.exp(mean)*np.exp(2*mean + variance**2)*(np.exp(variance**2)-1), dvariance_dx

        # For the exponentiated case, we assume that the variance is under control in general
        # This means that we expect the exponentiated variance gradient to be the same
        # as the gradient of the mean

        exp_dvariance_dx = mean*np.exp(mean)*dmean_dx #naive chain rule
        return variance, exp_dvariance_dx


    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)
