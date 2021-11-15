from plotting_common import cli
from nice_plots_paper import main as main_nice_plots_paper
from nice_plots import main as main_nice_plots
from tqdm import tqdm

TRANS_XLx4 = "transformer_xl,transformer_xl,transformer_xl,transformer_xl"
TRANENCx4 = (
    "transformer_encoder,transformer_encoder,transformer_encoder,transformer_encoder"
)
RESNET50x4 = "resnet50,resnet50,resnet50,resnet50"
BERTx4 = "bert_base_pretrained,bert_base_pretrained,bert_base_pretrained,bert_base_pretrained"
RESNET34x4 = "resnet34,resnet34,resnet34,resnet34"
LENETx4 = "lenet5,lenet5,lenet5,lenet5"

MNISTx4 = "mnist,mnist,mnist,mnist"
CIFAR10x4 = "cifar10,cifar10,cifar10,cifar10"
CIFAR100x4 = "cifar100,cifar100,cifar100,cifar100"
PTBx4 = "ptb,ptb,ptb,ptb"
WIKIx4 = "wikitext2,wikitext2,wikitext2,wikitext2"
SQUADx4 = "squad,squad,squad,squad"

ACCSTEP_BERT = "--acc_step=32,64,128,256"
BS_LENET = "--batch_size=4096,8192,16384,60000"
BS_RESNETS = "--batch_size=2048,4096,8192,16384"
BS_TRANENC = "--batch_size=256,256,256,256"
ACST_TRANENC = "--acc_step=8,16,32,232"
BS_TRANSXL = "--batch_size=64,64,64,64"
ACST_TRANSXL = "--acc_step=8,16,32,112"
BS_BERT = "--batch_size=24,24,24,24"

TR_LOSS = "--metric=training_loss"
TR_METRIC = "--metric=train_metric"

PLT_BEST = "--plot_type=best_run"
PLT_SS = "--plot_type=vs_ss"

MODELS_VISION = "lenet5,resnet34,resnet50"
MODELS_NLP = "transformer_encoder,transformer_xl,bert_base_pretrained"
MODELS = MODELS_VISION + "," + MODELS_NLP
DATASETS = "mnist,cifar10,cifar100,wikitext2,ptb,squad"

BATCH_SMALL = "--batch_size=128,128,128,64,64,24"
BATCH_FULL = "--batch_size=60000,20000,17000,256,64,24"
ACST_FULL = "--acc_step=-1,-1,-1,232,112,256"

F_BIG = "--big_batch"
F_FULL = "--full_batch"
F_MOM = "--momentum"

calls_nice_plots = [
    ###
    # From gen_big_batch_vs_ss.sh
    [LENETx4, MNISTx4, BS_LENET, PLT_SS, TR_LOSS, F_BIG],
    [LENETx4, MNISTx4, BS_LENET, PLT_SS, TR_METRIC, F_BIG],
    [RESNET34x4, CIFAR10x4, BS_RESNETS, PLT_SS, TR_LOSS, F_BIG],
    [RESNET34x4, CIFAR10x4, BS_RESNETS, PLT_SS, TR_METRIC, F_BIG],
    [RESNET50x4, CIFAR100x4, BS_RESNETS, PLT_SS, TR_LOSS, F_BIG],
    [RESNET50x4, CIFAR100x4, BS_RESNETS, PLT_SS, TR_METRIC, F_BIG],
    [TRANENCx4, WIKIx4, BS_TRANENC, ACST_TRANENC, PLT_SS, TR_LOSS, F_BIG],
    [TRANENCx4, WIKIx4, BS_TRANENC, ACST_TRANENC, PLT_SS, TR_METRIC, F_BIG],
    [TRANS_XLx4, PTBx4, BS_TRANSXL, ACST_TRANSXL, PLT_SS, TR_LOSS, F_BIG],
    [TRANS_XLx4, PTBx4, BS_TRANSXL, ACST_TRANSXL, PLT_SS, TR_METRIC, F_BIG],
    [BERTx4, SQUADx4, BS_BERT, ACCSTEP_BERT, PLT_SS, TR_LOSS, F_BIG],
    [BERTx4, SQUADx4, BS_BERT, ACCSTEP_BERT, PLT_SS, TR_METRIC, F_BIG],
    ###
    # From gen_big_batch.sh
    [LENETx4, MNISTx4, BS_LENET, PLT_BEST, TR_LOSS, F_BIG],
    [LENETx4, MNISTx4, BS_LENET, PLT_BEST, TR_METRIC, F_BIG],
    [RESNET34x4, CIFAR10x4, BS_RESNETS, PLT_BEST, TR_LOSS, F_BIG],
    [RESNET34x4, CIFAR10x4, BS_RESNETS, PLT_BEST, TR_METRIC, F_BIG],
    [RESNET50x4, CIFAR100x4, BS_RESNETS, PLT_BEST, TR_LOSS, F_BIG],
    [RESNET50x4, CIFAR100x4, BS_RESNETS, PLT_BEST, TR_METRIC, F_BIG],
    [TRANENCx4, WIKIx4, BS_TRANENC, ACST_TRANENC, PLT_BEST, TR_LOSS, F_BIG],
    [TRANENCx4, WIKIx4, BS_TRANENC, ACST_TRANENC, PLT_BEST, TR_METRIC, F_BIG],
    [TRANS_XLx4, PTBx4, BS_TRANSXL, ACST_TRANSXL, PLT_BEST, TR_LOSS, F_BIG],
    [TRANS_XLx4, PTBx4, BS_TRANSXL, ACST_TRANSXL, PLT_BEST, TR_METRIC, F_BIG],
    [BERTx4, SQUADx4, BS_BERT, ACCSTEP_BERT, PLT_BEST, TR_LOSS, F_BIG],
    [BERTx4, SQUADx4, BS_BERT, ACCSTEP_BERT, PLT_BEST, TR_METRIC, F_BIG],
]

call_standard_run_loss = [MODELS, DATASETS, BATCH_SMALL, PLT_BEST, TR_LOSS]
call_standard_run_metric = [MODELS, DATASETS, BATCH_SMALL, PLT_BEST, TR_METRIC]

calls_nice_plots_paper = [
    ###
    # From gen_plots.sh
    call_standard_run_loss,
    call_standard_run_metric,
    [MODELS, DATASETS, BATCH_FULL, ACST_FULL, PLT_BEST, TR_LOSS, F_FULL],
    [MODELS, DATASETS, BATCH_FULL, ACST_FULL, PLT_BEST, TR_METRIC, F_FULL],
    [MODELS, DATASETS, BATCH_SMALL, PLT_SS, TR_LOSS],
    [MODELS, DATASETS, BATCH_SMALL, PLT_SS, TR_METRIC],
    [MODELS, DATASETS, BATCH_SMALL, PLT_SS, TR_LOSS, F_MOM],
    [MODELS, DATASETS, BATCH_SMALL, PLT_SS, TR_METRIC, F_MOM],
    [MODELS, DATASETS, BATCH_FULL, ACST_FULL, PLT_SS, TR_LOSS, F_FULL],
    [MODELS, DATASETS, BATCH_FULL, ACST_FULL, PLT_SS, TR_METRIC, F_FULL],
    [MODELS, DATASETS, BATCH_FULL, ACST_FULL, PLT_SS, TR_LOSS, F_FULL, F_MOM],
    [MODELS, DATASETS, BATCH_FULL, ACST_FULL, PLT_SS, TR_METRIC, F_FULL, F_MOM],
]

if __name__ == "__main__":
    for call in tqdm(calls_nice_plots_paper):
        print("# python nice_plot_paper.py " + " ".join(call))
        main_nice_plots_paper(cli().parse_args(call))
    for call in tqdm(calls_nice_plots):
        print("# python nice_plot.py " + " ".join(call))
        main_nice_plots(cli().parse_args(call))
