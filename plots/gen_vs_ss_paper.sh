#!/bin/bash

python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=128,128,128,64,64,24 --plot_type=vs_ss --metric=train_metric --momentum
python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=128,128,128,64,64,24 --plot_type=vs_ss --metric=train_metric
python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=60000,20000,17000,1300,500,24 --plot_type=vs_ss --metric=train_metric --full_batch
python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=60000,20000,17000,1300,500,24 --plot_type=vs_ss --metric=train_metric --full_batch --momentum
