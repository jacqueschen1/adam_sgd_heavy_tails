#!/bin/bash
python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=60000,20000,17000,1300,500,24 --plot_type=best_run --metric=training_loss --full_batch
python plots_for_paper.py lenet5,resnet34,resnet50,transformer_encoder,transformer_xl,bert_base_pretrained mnist,cifar10,cifar100,wikitext2,ptb,squad --timestamp=1624365143 --batch_size=60000,20000,17000,1300,500,24 --plot_type=best_run --metric=train_metric --full_batch
