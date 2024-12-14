import os
import sys
import time
import wandb
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from funcsigcls import FuncSigCls
from eklavyadataset import EklavyaDataset
sys.path.insert(0,"src/utils")

from save_model import save_model
from yaml_config_hook import yaml_config_hook
from metrics import calculate_metrics, results_to_excel

key_to_pop = ['opti','compiler']

def train(args, loader, model, type_criterion, count_criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.train()
    for step, (x, (y, y_cnt)) in enumerate(loader):
        optimizer.zero_grad()

        x_info = {}
        for key in key_to_pop:
            x_info[key] = x.pop(key)

        x = {feature: value.cuda(non_blocking=True) for feature, value in x.items()}
        y = y.to(args.device)
        y_cnt = y_cnt.to(args.device)

        arg_types_preds, arg_cnt_preds = model(x)

        loss = 0
        for i in range(args.num_args):
            loss += type_criterion(arg_types_preds[i], y[:,i])
        loss /= args.num_args

        loss += count_criterion(arg_cnt_preds, y_cnt)

        train_acc = []
        for i in range(args.num_args):
            train_acc.append((arg_types_preds[i].argmax(dim=-1) == y[:,i]).sum().item() / y.size(0))
        
        train_acc.append((arg_cnt_preds.argmax(dim=-1) == y_cnt).sum().item() / y.size(0))
        train_acc = sum(train_acc)/len(train_acc)

        accuracy_epoch += train_acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        if step % args.show_step == 0:
            print(
                 f"Train Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {train_acc}"
             )

    return loss_epoch, accuracy_epoch


def test(args, loader, model, type_criterion, count_criterion):
    model.eval()
    loss_epoch = 0
    correct_predictions_type = {}
    total_predictions_type = {}
    false_positives_type = {}
    false_negatives_type = {}
    correct_predictions_count = {}
    total_predictions_count = {}
    false_positives_count = {}
    false_negatives_count = {}
    total_inference_time = 0
    total_samples = 0

    for step, (x, (y, y_cnt)) in enumerate(loader):
        model.zero_grad()

        x_info = {}
        for key in key_to_pop:
            x_info[key] = x.pop(key)

        x = {feature: value.cuda(non_blocking=True) for feature, value in x.items()}
        y = y.to(args.device)
        y_cnt = y_cnt.to(args.device)

        total_samples += y.shape[0]

        with torch.no_grad():
            start_time = time.time()
            arg_types_preds, arg_cnt_preds = model(x)

            loss = 0
            for i in range(args.num_args):
                loss += type_criterion(arg_types_preds[i], y[:,i])
            loss /= args.num_args

            loss += count_criterion(arg_cnt_preds, y_cnt)

            actual_type_predictions = []
            for i in range(args.num_args):
                actual_type_predictions.append(arg_types_preds[i].argmax(dim=-1))
            actual_count_predictions = arg_cnt_preds.argmax(dim=-1)
            archs = args.arch_tokenizer.convert_ids_to_tokens(x['arch'][:, 0])
            optis = x_info['opti']
            compilers = x_info['compiler']

            for i in range(args.num_args):
                for comp, arch, pred, label in zip(compilers, archs, actual_type_predictions[i], y[:,i]):

                    if isinstance(model, torch.nn.DataParallel):
                        pred, label = model.module.model.config.id2label[pred.item()], model.module.model.config.id2label[label.item()]
                    else:
                        pred, label = model.model.config.id2label[pred.item()], model.model.config.id2label[label.item()]
                    
                    total_predictions_type[comp, arch, label] = total_predictions_type.get((comp, arch, label), 0) + 1

                    if pred == label:
                        correct_predictions_type[comp, arch, label] = correct_predictions_type.get((comp, arch, label), 0) + 1
                    
                    if pred != label:
                        false_positives_type[comp, arch, pred] = false_positives_type.get((comp, arch, pred), 0) + 1
                        false_negatives_type[comp, arch, label] = false_negatives_type.get((comp, arch, label), 0) + 1
                    
            for comp, arch, pred, label in zip(compilers, archs, actual_count_predictions, y_cnt):
                pred, label = args.count_list[pred.item()], args.count_list[label.item()]

                total_predictions_count[comp, arch, label] = total_predictions_count.get((comp, arch, label), 0) + 1

                if pred == label:
                    correct_predictions_count[comp, arch, label] = correct_predictions_count.get((comp, arch, label), 0) + 1
                
                if pred != label:
                    false_positives_count[comp, arch, pred] = false_positives_count.get((comp, arch, pred), 0) + 1
                    false_negatives_count[comp, arch, label] = false_negatives_count.get((comp, arch, label), 0) + 1
            
            loss_epoch += loss.item()

            if step % args.show_step == 0:
                print(
                    f"Test Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t"
                )

            end_time = time.time()
            total_inference_time += (end_time - start_time)

    average_inference_time_per_sample = total_inference_time / total_samples

    return loss_epoch, correct_predictions_type, total_predictions_type, false_positives_type, false_negatives_type, correct_predictions_count, total_predictions_count, false_positives_count, false_negatives_count, average_inference_time_per_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("src/0_environment_setup/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_size = None
    if ('95_5' in args.model_path) or ('6_reload_then_finetune' in args.model_path):
        assert args.model_path.split('/')[-1] != '95_5', 'need data size'
        data_size, dataset_split, model_size  = args.model_path.split('/')[-1], args.model_path.split('/')[-2], args.model_path.split('/')[-3]
    else:
        assert 'per' not in args.model_path.split('/')[-1], 'no need data size'
        dataset_split, model_size  = args.model_path.split('/')[-1], args.model_path.split('/')[-2]

    if args.dataset_arch == 'obf':
        args.dataset_pt = f'workdir/4_prepare_finetune_dataset/obf/funcsig/{dataset_split}'
        projectname = f"GBME_funcsig_{dataset_split}_{args.backbone.lower()}_{model_size}_obf"
    else:
        raise Exception('wrong dataset')

    if args.backbone.lower() == 'roberta':
        if model_size == '1GB':
            assert (args.projection_dim == 768) and (args.layers == 12), 'wrong model configuration'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    finetune_from = 'funcsim'
    pretrain_split = '95_5'

    name_list = ['funcsig', args.backbone.lower()]
    if data_size is not None:
        if '95_5' in args.dataset_pt:
            args.dataset_pt = os.path.join(args.dataset_pt, data_size)
        name_list.append('data_size')
        name_list.append(data_size)
    print(f"dataset path is : {args.dataset_pt}")

    if '6_reload_then_finetune' in args.model_path:
        name_list.insert(1, 'finetuned')
        name_list.append('pretraintask')
        name_list.append(finetune_from)
        name_list.append('pretraindatasplit')
        name_list.append(pretrain_split)
        name_list.append('checkpoint')
        name_list.append(str(args.epoch_num))
    elif '5_finetune_linear_evaluation' in args.model_path:
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

    train_dataset = EklavyaDataset(args, 'train')
    test_dataset = EklavyaDataset(args, 'test')

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
    assert 'funcsig' in args.model_path, 'dataset and path not compatible'
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

    model = FuncSigCls(args)
    if args.reload:
        print('Reloading...')
        if '3_pretrain_contrastive_learning' in args.model_path:
            saved_model_pt = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
            checkpoint = torch.load(saved_model_pt)
            print(f'Reload from pretraining at {saved_model_pt}.....')
            model.load_state_dict(checkpoint['model_state_dict'])
        elif '5_finetune_linear_evaluation' in args.model_path:
            saved_model_pt = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
            checkpoint = torch.load(saved_model_pt)
            print(f'Continue finetuning at {saved_model_pt}.....')
            args.logistic_start_epoch = checkpoint['epoch']
            assert args.logistic_start_epoch < args.logistic_epochs, 'invalid logistic_start_epoch'
            model.load_state_dict(checkpoint['model_state_dict'])
        elif '6_reload_then_finetune' in args.model_path:
            saved_model_pt = os.path.join(f'workdir/5_finetune_linear_evaluation/x64/{finetune_from}',args.backbone.lower(),model_size,f'{pretrain_split}',data_size,f'checkpoint_{args.epoch_num}.tar')
            model_state_dict = model.state_dict()
            checkpoint_state_dict = torch.load(saved_model_pt)['model_state_dict']
            print(f'[Finetune] from previous task at {saved_model_pt} ...') 
            common_keys = model_state_dict.keys() & checkpoint_state_dict.keys()
            missing_keys = model_state_dict.keys() - checkpoint_state_dict.keys()
            unexpected_keys = checkpoint_state_dict.keys() - model_state_dict.keys()
            print("Common keys:", common_keys)
            print("Missing keys in checkpoint:", missing_keys)
            print("Unexpected keys in checkpoint:", unexpected_keys)
            model.load_state_dict(checkpoint_state_dict, strict=False)
        else:
            raise Exception('Invalid path')
    else:
        if args.MLM_pretrain:
            if '95_5' in args.model_path:
                saved_model_pt = os.path.join(f'workdir/5_finetune_linear_evaluation/x64/{finetune_from}',args.backbone.lower(),model_size,f'{pretrain_split}',data_size,f'checkpoint_{args.epoch_num}.tar')
                print(f'[Multi-stage Transfer] from previous task at {saved_model_pt} ...') 
            else:
                saved_model_pt = os.path.join(f'workdir/4.5_mlm_task/x64',args.backbone.lower(),model_size,f'{pretrain_split}','5k',f'checkpoint_{args.MLM_pretrain_baseline_epoch}.tar')
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
    type_criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    count_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    print('Start finetuning ...')
    args.current_epoch = args.logistic_start_epoch 

    # Early stopping parameters
    best_f1_score_type = 0.0
    best_f1_score_count = 0.0
    best_f1_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = args.early_stopping_patience
    
    for epoch in range(args.logistic_start_epoch, args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(
            args, train_loader, model, type_criterion, count_criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Acc:{accuracy_epoch / len(train_loader)}]"
        )
        args.current_epoch += 1

        if (epoch+1) % args.save_epoch == 0:
            save_model(args, model, optimizer)

        if (epoch+1) % args.validation_epoch == 0:
            print("Start Validation ...")
            loss_epoch, correct_predictions_type, total_predictions_type, false_positives_type, false_negatives_type, correct_predictions_count, total_predictions_count, false_positives_count, false_negatives_count, average_inference_time = test(
                args, test_loader, model, type_criterion, count_criterion
            )

            metrics_type = calculate_metrics(correct_predictions_type, total_predictions_type, false_positives_type, false_negatives_type, args.label_list)
            metrics_count = calculate_metrics(correct_predictions_count, total_predictions_count, false_positives_count, false_negatives_count, args.count_list)
            
            #tensor_to_float(metrics)
            '''
            for elements, reports in metrics.items():
                print(f"[Average: {elements}]\t Precision: {reports['Average']['Precision']}\t Recall: {reports['Average']['Recall']}\t F1 Score: {reports['Average']['F1 Score']}\t Accuracy:{reports['Average']['Accuracy']}")
                
                for class_label in args.label_list:
                    print(f"[CLASS: {class_label}]\t Data Distribution: {reports[class_label]['Weight']}\t Precision: {reports[class_label]['Precision']}\t Recall: {reports[class_label]['Recall']}\t F1 Score: {reports[class_label]['F1 Score']}\t Accuracy:{reports[class_label]['Accuracy']}")
            '''
            results_to_excel(metrics_type, os.path.join(args.model_path, 'output_type.xlsx'), show_class=True)
            results_to_excel(metrics_count, os.path.join(args.model_path, 'output_count.xlsx'), show_class=True)

            current_f1_score_type = sum(reports['Average']['F1 Score'] for elements, reports in metrics_type.items()) / len(metrics_type)
            current_f1_score_count = sum(reports['Average']['F1 Score'] for elements, reports in metrics_count.items()) / len(metrics_count)

            # Early stopping check
            if current_f1_score_type > best_f1_score_type or current_f1_score_count > best_f1_score_count:
                best_f1_score_type = current_f1_score_type
                best_f1_score_count = current_f1_score_count
                best_f1_epoch = args.current_epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            wandb.log({
                'epoch': args.current_epoch,
                'f1_score_type': current_f1_score_type,
                'f1_score_count': current_f1_score_count,
                'inference_time': average_inference_time,
                })

            # Check if early stopping is triggered
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered. Stopping training at epoch {args.current_epoch}.")
                break 

    wandb.log({
        'best_f1_score_type': best_f1_score_type,
        'best_f1_score_count': best_f1_score_count,
        'best_f1_epoch': best_f1_epoch
        })
                    
    save_model(args, model, optimizer)
    wandb.finish()
