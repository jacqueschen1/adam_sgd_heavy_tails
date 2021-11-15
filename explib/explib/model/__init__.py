from .letnet5 import LeNet5
from .linear_model import LinearModel
from .transformer_encoder import TransformerEncoderModel
from .resnet import getResNet
from .full_connected import FullyConnected
from .transformer_xl import MemTransformerLM
from .sls_models import get_model
from .bert_base_pretrained import get_bert_base_pretrained
from ..util import weights_init

AVAILABLE_MODELS = [
    "lenet5",
    "lin_reg",
    "log_reg",
    "transformer_encoder",
    "resnet50",
    "resnet34",
    "resnet101",
    "fc",
    "transformer_xl",
    "sls_resnet50",
    "sls_resnet34",
    "sls_resnet18",
    "bert_base_pretrained",
]


def init(model_name, model_args=None, features_dim=0):
    if model_name not in AVAILABLE_MODELS:
        raise Exception("Model {} not available".format(model_name))

    if model_name == "lenet5":
        if model_args is not None:
            return LeNet5(10, in_channels=model_args["in_channels"])
        return LeNet5(10)

    if model_name == "lin_reg":
        return LinearModel(features_dim, 1)

    if model_name == "log_reg":
        return LinearModel(features_dim, 2)

    if model_name == "transformer_encoder":
        model = TransformerEncoderModel(features_dim, 200, 2, 200, 2, 0.2)
        model.apply(weights_init)
        return model

    if model_name == "resnet50":
        return getResNet(50, model_args=model_args)

    if model_name == "resnet34":
        return getResNet(34, model_args=model_args)

    if model_name == "resnet101":
        return getResNet(101)

    if model_name.startswith("sls_"):
        return get_model(model_name[4:] + "_100", model_args)

    if model_name == "fc":
        return FullyConnected()

    if model_name == "transformer_xl":
        model = MemTransformerLM(
            features_dim,
            model_args["n_layer"],
            model_args["n_head"],
            model_args["d_model"],
            model_args["d_head"],
            model_args["d_inner"],
            model_args["dropout"],
            model_args["dropatt"],
            tie_weight=False,
            d_embed=model_args["d_model"],
            tgt_len=model_args["tgt_len"],
            ext_len=0,
            mem_len=model_args["mem_len"],
            same_length=False,
        )
        model.apply(weights_init)
        model.word_emb.apply(weights_init)
        return model

    if model_name == "bert_base_pretrained":
        return get_bert_base_pretrained()
