from models.ae import Autoencoder
from models.determinstic_ebm import DeterminsticEBM
from models.latent_minimization_ebm import LatentMinimizationEBM
from models.linear_predictor import LinearPredictor
from models.test_model import TestModel

model_dict = {
    "ebm": LatentMinimizationEBM,
    "gen": DeterminsticEBM,
    "ae": Autoencoder,
    "linpred": LinearPredictor,
    "test": TestModel,
}
