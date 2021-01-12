from models.components.encoder import *
from models.components.decoder import *
from models.components.predictor import *


encoder_dict = {
    "dummy": DummyFrameEncoder,
    "resnet": ResNetFrameEncoder,
    "test": TestFrameEncoder,
    "test2": TestFrameEncoder2,
}

decoder_dict = {
    "dummy": DummyFrameDecoder,
    "resnet": ResNetFrameDecoder,
    "test": TestFrameDecoder,
    "test2": TestFrameDecoder2,
}

hidden_predictor_dict = {
    "transformer_no_latent": TransformerHiddenPredictor_no_latent,
    "transformer_no_latent_3d": TransformerHiddenPredictor_no_latent_3D,
    "transformer1": TransformerHiddenPredictor1,
    "transformer1_ptp": TransformerHiddenPredictor1_ptp,
    "transformer1_3d": TransformerHiddenPredictor1_3D,
    "transformer1_ptp_3d": TransformerHiddenPredictor1_ptp_3D,
    "transformer2": TransformerHiddenPredictor2,
    "transformer2_ptp": TransformerHiddenPredictor2_ptp,
    "transformer2_3d": TransformerHiddenPredictor2_3D,
    "transformer2_ptp_3d": TransformerHiddenPredictor2_ptp_3D,
    "transformer3": TransformerHiddenPredictor3,
    "add_no_latent": AddHiddenPredictor_no_latent,
    "add": AddHiddenPredictor,
    "baseline_no_latent": BaselineHiddenPredictor_no_latent,
    "baseline": BaselineHiddenPredictor,
    "baseline_ptp": BaselineHiddenPredictor_ptp,
}
