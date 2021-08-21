#!/bin/bash

# For paper
python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=128,128,128,64,64,24 --plot_type=best_run --metric=training_loss
python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=128,128,128,64,64,24 --plot_type=best_run --metric=train_metric
