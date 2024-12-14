import os
import sys
import time
import torch
import wandb
import argparse
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from instboundcls import InstBoundcls
from specdataset import SpecDataset
sys.path.insert(0,"src/utils")

from save_model import save_model
from yaml_config_hook import yaml_config_hook
from metrics import calculate_metrics, results_to_excel

key_to_pop = ['opti', 'package', 'compiler']

def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    f1_epoch = 0
    model.train()
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x_info = {}
        for key in key_to_pop:
            x_info[key] = x.pop(key)

        x = {feature: value.cuda(non_blocking=True) for feature, value in x.items()}
        y = y.to(args.device)

        z = model(x)

        z = z.reshape(-1, 2)
        y = y.reshape(-1)
        loss = criterion(z, y)           

        z = torch.nn.functional.softmax(z, dim=-1)
        y_true_cpu = y.cpu().numpy()
        y_pred_cpu = z.argmax(dim=-1).cpu().numpy()

        mask = y_true_cpu != -100
        filtered_true = y_true_cpu[mask]
        filtered_pred = y_pred_cpu[mask]

        f1_step = f1_score(filtered_true, filtered_pred, average='micro')
        f1_epoch += f1_step

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        if step % args.show_step == 0:
            print(
                 f"Train Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t F1:{f1_step}"
             )

    return loss_epoch, f1_epoch


def test(args, loader, model, criterion):
    loss_epoch = 0
    #f1_epoch = 0
    correct_predictions = {}
    total_predictions = {}
    false_positives = {}
    false_negatives = {}
    total_inference_time = 0
    total_samples = 0

    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x_info = {}
        for key in key_to_pop:
            x_info[key] = x.pop(key)

        x = {feature: value.cuda(non_blocking=True) for feature, value in x.items()}
        y = y.to(args.device)

        total_samples += y.shape[0]

        with torch.no_grad():
            start_time = time.time()
            z = model(x)

            z = z.reshape(-1, 2)
            y = y.reshape(-1)

            loss = criterion(z, y)
            z = torch.nn.functional.softmax(z, dim=-1)

            y_true_cpu = y.cpu().numpy()
            y_pred_cpu = z.argmax(dim=-1).cpu().numpy()

            mask = y_true_cpu != -100
            filtered_true = y_true_cpu[mask]
            filtered_pred = y_pred_cpu[mask]

            archs = args.arch_tokenizer.convert_ids_to_tokens(x['arch'][:, 0])
            optis = x_info['opti']
            #packages = x_info['package']
            #platforms = x_info['platform']
            compilers = x_info['compiler']

            for opti, compiler, arch, pred, label in zip(optis, compilers, archs, filtered_pred, filtered_true):

                if isinstance(model, torch.nn.DataParallel):
                    pred, label = model.module.model.config.id2label[pred.item()], model.module.model.config.id2label[label.item()]
                else:
                    pred, label = model.model.config.id2label[pred.item()], model.model.config.id2label[label.item()]

                total_predictions[opti, compiler, arch, label] = total_predictions.get((opti, compiler, arch, label), 0) + 1

                if pred == label:
                    correct_predictions[opti, compiler, arch, label] = correct_predictions.get((opti, compiler, arch, label), 0) + 1

                if pred != label:
                    false_positives[opti, compiler, arch, pred] = false_positives.get((opti, compiler, arch, pred), 0) + 1
                    false_negatives[opti, compiler, arch, label] = false_negatives.get((opti, compiler, arch, label), 0) + 1

            loss_epoch += loss.item()

            if step % args.show_step == 0:
                print(
                    f"Test Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t"
                )

            end_time = time.time()
            total_inference_time += (end_time - start_time)

    average_inference_time_per_sample = total_inference_time / total_samples

    return loss_epoch, correct_predictions, total_predictions, false_positives, false_negatives, average_inference_time_per_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("src/0_environment_setup/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_size = None
    if '95_5' in args.model_path:
        assert args.model_path.split('/')[-1] != '95_5', 'need data size'
        data_size, dataset_split, model_size  = args.model_path.split('/')[-1], args.model_path.split('/')[-2], args.model_path.split('/')[-3]
    else:
        assert 'per' not in args.model_path.split('/')[-1], 'no need data size'
        dataset_split, model_size  = args.model_path.split('/')[-1], args.model_path.split('/')[-2]

    if args.dataset_arch == 'obf':
        args.dataset_pt = f'workdir/4_prepare_finetune_dataset/obf/instbound/{args.input_length}/{dataset_split}'
        projectname = f"GBME_instbound_{dataset_split}_{args.backbone.lower()}_{model_size}_obf"
    else:
        raise Exception('wrong dataset')

    if args.backbone.lower() == 'roberta':
        if model_size == '1GB':
            assert (args.projection_dim == 768) and (args.layers == 12), 'wrong model configuration'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    name_list = ['instbound', args.backbone.lower()]

    if data_size is not None:
        if '95_5' in args.dataset_pt:
            args.dataset_pt = os.path.join(args.dataset_pt, data_size)
        name_list.append('data_size')
        name_list.append(data_size)
    print(f"dataset path is : {args.dataset_pt}")

    if '95_5' not in args.model_path:
        if not args.MLM_pretrain:
            name_list.insert(1, 'no_mlm_pretrained')
            args.model_path = os.path.join(args.model_path, 'no_mlm')
        else:
            args.model_path = os.path.join(args.model_path, 'with_mlm')
    
    os.environ['WANDB_DIR'] = args.model_path
    wandb.init(project=projectname, entity='lhxxh', name='_'.join(name_list))
    wandb.config.batch_size = args.logistic_batch_size
    wandb.config.learning_rate = 1e-5

    print('Supervised finetune arguments')
    for k, v in vars(args).items():
        print(f'{k} : {v}')

    train_dataset = SpecDataset(args, 'train')
    test_dataset = SpecDataset(args, 'test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,   
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    # load pre-trained model from checkpoint 
    assert 'instbound' in args.model_path, 'dataset and path not compatible'
    if not args.reload:
        assert '5_finetune_linear_evaluation' in args.model_path, 'incorrect model path'
    if args.backbone == 'RoBerta':
        assert 'roberta' in args.model_path, 'backbone and path not compatible'
    elif args.backbone == 'Longformer':
        assert 'longformer' in args.model_path, 'backbone and path not compatible'
    elif args.backbone == 'MalConv2':
        assert 'malconv2' in args.model_path, 'backbone and path not compatible'
    else:
        raise NotImplementedError

    model = InstBoundcls(args)
    if args.reload:
        print('Reloading...')
        saved_model_pt = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        checkpoint = torch.load(saved_model_pt)
        if '3_pretrain_contrastive_learning' in args.model_path:
            print(f'Reload from pretraining at {saved_model_pt}.....')
            model.load_state_dict(checkpoint['model_state_dict'])
        elif '5_finetune_linear_evaluation' in args.model_path:
            print(f'Continue finetuning at {saved_model_pt}.....')
            args.logistic_start_epoch = checkpoint['epoch']
            assert args.logistic_start_epoch < args.logistic_epochs, 'invalid logistic_start_epoch'
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise Exception('Invalid path')
    else:
        if args.MLM_pretrain:
            saved_model_pt = os.path.join(f'workdir/4.5_mlm_task/x64/',args.backbone.lower(),model_size,'95_5','5k',f'checkpoint_{args.MLM_pretrain_baseline_epoch}.tar')
            if '95_5' in args.model_path:
                print(f'[Multi-stage Transfer] from previous task at {saved_model_pt} ...') 
            else:
                print(f'[Baseline] MLM pretrain path at {saved_model_pt} ...')
            model_state_dict = model.state_dict()
            checkpoint_state_dict = torch.load(saved_model_pt)['model_state_dict']
            # For malconv
            checkpoint_state_dict.pop('classifier.weight', None)
            checkpoint_state_dict.pop('classifier.bias', None)
            # For roberta
            checkpoint_state_dict.pop('model.classifier.weight', None)
            checkpoint_state_dict.pop('model.classifier.bias', None)      
            common_keys = model_state_dict.keys() & checkpoint_state_dict.keys()
            missing_keys = model_state_dict.keys() - checkpoint_state_dict.keys()
            unexpected_keys = checkpoint_state_dict.keys() - model_state_dict.keys()
            print("Common keys:", common_keys)
            print("Missing keys in checkpoint:", missing_keys)
            print("Unexpected keys in checkpoint:", unexpected_keys)
            model.load_state_dict(checkpoint_state_dict, strict=False)

    if torch.cuda.device_count() > 1:
        print(f'Use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay = args.weight_decay)
    if args.reload and '5_finetune_linear_evaluation' in args.model_path:
        print('Reuse previous optimizer..')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    print('Start finetuning ...')
    args.current_epoch = args.logistic_start_epoch 

    # Early stopping parameters
    best_f1_score = 0.0
    best_f1_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = args.early_stopping_patience

    for epoch in range(args.logistic_start_epoch, args.logistic_epochs):
        loss_epoch, f1_epoch = train(
            args, train_loader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t F1: {f1_epoch / len(train_loader)}"
        )
        args.current_epoch += 1

        if (epoch+1) % args.save_epoch == 0:
            save_model(args, model, optimizer)

        if (epoch+1) % args.validation_epoch == 0:
            print("Start Validation ...")
            loss_epoch, correct_predictions, total_predictions, false_positives, false_negatives, average_inference_time = test(
                args, test_loader, model, criterion
            )
            
            metrics = calculate_metrics(correct_predictions, total_predictions, false_positives, false_negatives, args.label_list)

            '''
            for elements, reports in metrics.items():
                print(f"[Architecture: {elements}]\t Precision: {reports['Average']['Precision']}\t Recall: {reports['Average']['Recall']}\t F1 Score: {reports['Average']['F1 Score']}\t Accuracy:{reports['Average']['Accuracy']}")

                for class_label in args.label_list:
                    print(f"[CLASS: {class_label}]\t Data Distribution: {reports[class_label]['Weight']}\t Precision: {reports[class_label]['Precision']}\t Recall: {reports[class_label]['Recall']}\t F1 Score: {reports[class_label]['F1 Score']}\t Accuracy:{reports[class_label]['Accuracy']}")
            '''
            results_to_excel(metrics, os.path.join(args.model_path, 'output.xlsx'), show_class=True)

            average_f1_score = sum(reports['Average']['F1 Score'] for elements, reports in metrics.items()) / len(metrics)

            # Early stopping check
            if average_f1_score > best_f1_score:
                best_f1_score = average_f1_score
                best_f1_epoch = args.current_epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            wandb.log({
                'epoch': args.current_epoch,
                'f1_score': average_f1_score,
                'inference_time': average_inference_time,
                })

            # Check if early stopping is triggered
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered. Stopping training at epoch {args.current_epoch}.")
                break 

    wandb.log({
        'best_f1_score': best_f1_score,
        'best_f1_epoch': best_f1_epoch
        })

    save_model(args, model, optimizer)
    wandb.finish()