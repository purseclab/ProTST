import os
import sys
import time
import torch
import wandb
import argparse
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from mlmcls import MLMcls
from mlmdataset import MLMDataset
sys.path.insert(0,"src/utils")

from save_model import save_model
from yaml_config_hook import yaml_config_hook
from metrics import calculate_metrics, results_to_excel

def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    f1_epoch = 0

    model.train()
    for step, (x, y, indices_to_mask) in enumerate(loader):
        optimizer.zero_grad()

        x = {feature: value.cuda(non_blocking=True) for feature, value in x.items()}
        y = y.to(args.device)

        z = model(x)

        indices_to_mask = indices_to_mask.to(args.device)
        selected_logits = torch.gather(z, dim=1, index=indices_to_mask.unsqueeze(-1).expand(-1, -1, z.size(-1)))
        probs = F.softmax(selected_logits, dim=-1)

        y_selected = torch.gather(y, dim=1, index=indices_to_mask)
        target_probs = torch.gather(probs, dim=-1, index=y_selected.unsqueeze(-1))
        ppl = torch.exp(-torch.log(target_probs).mean())

        z = z.reshape(-1, 256)
        y = y.reshape(-1)
        loss = criterion(z, y)       

        z = torch.nn.functional.softmax(z, dim=-1)
        y_true_cpu = y.cpu().numpy()
        y_pred_cpu = z.argmax(dim=-1).cpu().numpy()

        f1_step = f1_score(y_true_cpu, y_pred_cpu, average='micro')
        f1_epoch += f1_step

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        if step % args.show_step == 0:
            print(
                 f"Train Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t F1:{f1_step}\t PPL:{ppl}"
             )

    return loss_epoch, f1_epoch


def test(args, loader, model, criterion):
    loss_epoch = 0
    f1_epoch = 0
    ppl_epoch = 0

    model.eval()
    for step, (x, y, indices_to_mask) in enumerate(loader):
        model.zero_grad()

        x = {feature: value.cuda(non_blocking=True) for feature, value in x.items()}
        y = y.to(args.device)

        with torch.no_grad():
            start_time = time.time()
            z = model(x)

            indices_to_mask = indices_to_mask.to(args.device)
            selected_logits = torch.gather(z, dim=1, index=indices_to_mask.unsqueeze(-1).expand(-1, -1, z.size(-1)))
            probs = F.softmax(selected_logits, dim=-1)

            y_selected = torch.gather(y, dim=1, index=indices_to_mask)
            target_probs = torch.gather(probs, dim=-1, index=y_selected.unsqueeze(-1))
            ppl = torch.exp(-torch.log(target_probs).mean())
            ppl_epoch += ppl

            z = z.reshape(-1, 256)
            y = y.reshape(-1)

            loss = criterion(z, y)
            z = torch.nn.functional.softmax(z, dim=-1)

            y_true_cpu = y.cpu().numpy()
            y_pred_cpu = z.argmax(dim=-1).cpu().numpy()

            f1_step = f1_score(y_true_cpu, y_pred_cpu, average='micro')
            f1_epoch += f1_step

            loss_epoch += loss.item()

            if step % args.show_step == 0:
                print(
                    f"Test Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t PPL:{ppl}"
                )

    average_f1 = f1_epoch / (step + 1)
    average_ppl = ppl_epoch / (step + 1)

    return loss_epoch, average_f1, average_ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("src/0_environment_setup/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ['WANDB_DIR'] = args.model_path

    data_size, dataset_split, model_size  = args.model_path.split('/')[-1], args.model_path.split('/')[-2], args.model_path.split('/')[-3]
    assert dataset_split == '95_5', 'wrong dataset split'

    if args.dataset_arch == 'x86':
        args.dataset_pt = f'workdir/4_prepare_finetune_dataset/x86/mlm/{args.input_length}/{dataset_split}/{data_size}'
        projectname = f"GBME_mlm_{dataset_split}_{args.backbone.lower()}_{model_size}_x86_32"
    elif args.dataset_arch == 'all':
        args.dataset_pt = f'workdir/4_prepare_finetune_dataset/all_architecture/mlm/{args.input_length}/{dataset_split}/{data_size}'
        projectname = f"GBME_mlm_{dataset_split}_{args.backbone.lower()}_{model_size}_allarch"
    else:
        raise Exception('wrong dataset')
    print(f"dataset path is : {args.dataset_pt}")

    if args.backbone.lower() == 'roberta':
        if model_size == '1GB':
            assert (args.projection_dim == 768) and (args.layers == 12), 'wrong model configuration'
        elif model_size == '512MB':
            assert (args.projection_dim == 480) and (args.layers == 12), 'wrong model configuration'
        elif model_size == '256MB':
            assert (args.projection_dim == 264) and (args.layers == 12), 'wrong model configuration'
        else:
            raise NotImplementedError
    elif args.backbone.lower() == 'malconv2':
        if model_size == '100MB':
            assert (args.projection_dim == 320) and (args.layers == 12), 'wrong model configuration'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    name_list = ['mlm', args.backbone.lower(), 'data_size', data_size]
    wandb.init(project=projectname, entity='lhxxh', name='_'.join(name_list))
    wandb.config.batch_size = args.logistic_batch_size
    wandb.config.learning_rate = 1e-5

    print('Supervised finetune arguments')
    for k, v in vars(args).items():
        print(f'{k} : {v}')

    train_dataset = MLMDataset(args, 'train')
    test_dataset = MLMDataset(args, 'test')

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
    assert 'mlm' in args.model_path, 'dataset and path not compatible'
    if not args.reload:
        assert '4.5_mlm_task' in args.model_path, 'incorrect model path'
    if args.backbone == 'RoBerta':
        assert 'roberta' in args.model_path, 'backbone and path not compatible'
    elif args.backbone == 'Longformer':
        assert 'longformer' in args.model_path, 'backbone and path not compatible'
    elif args.backbone == 'MalConv2':
        assert 'malconv2' in args.model_path, 'backbone and path not compatible'
    else:
        raise NotImplementedError

    model = MLMcls(args)
    if args.reload:
        print('Reloading...')
        saved_model_pt = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        checkpoint = torch.load(saved_model_pt)
        print(f'Continue finetuning at {saved_model_pt}.....')
        args.logistic_start_epoch = checkpoint['epoch']
        assert args.logistic_start_epoch < args.logistic_epochs, 'invalid logistic_start_epoch'
        model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.device_count() > 1:
        print(f'Use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay = args.weight_decay)
    if args.reload:
        print('Reuse previous optimizer..')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = torch.nn.CrossEntropyLoss()

    print('Start finetuning ...')
    args.current_epoch = args.logistic_start_epoch 

    # Early stopping parameters
    best_f1_score = 0.0
    best_ppl_score = 1000.0
    best_epoch = 0
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

        if epoch % args.save_epoch == 0:
            save_model(args, model, optimizer)

        if epoch % args.validation_epoch == 0:
            print("Start Validation ...")
            loss_epoch, average_f1, average_ppl = test(
                args, test_loader, model, criterion
            )

            '''
            for elements, reports in metrics.items():
                print(f"[Architecture: {elements}]\t Precision: {reports['Average']['Precision']}\t Recall: {reports['Average']['Recall']}\t F1 Score: {reports['Average']['F1 Score']}\t Accuracy:{reports['Average']['Accuracy']}")

                for class_label in args.label_list:
                    print(f"[CLASS: {class_label}]\t Data Distribution: {reports[class_label]['Weight']}\t Precision: {reports[class_label]['Precision']}\t Recall: {reports[class_label]['Recall']}\t F1 Score: {reports[class_label]['F1 Score']}\t Accuracy:{reports[class_label]['Accuracy']}")
            '''

            # Early stopping check
            if average_f1 > best_f1_score or average_ppl < best_ppl_score:
                best_f1_score = average_f1
                best_ppl_score = average_ppl
                best_epoch = args.current_epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            print(f'[Validation]: {average_f1}')
            wandb.log({
                'epoch': args.current_epoch,
                'f1_score': average_f1,
                'PPL': average_ppl
                })

            # Check if early stopping is triggered
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered. Stopping training at epoch {args.current_epoch}.")
                break 
                
    wandb.log({
        'best_f1_score': best_f1_score,
        'best_ppl_score': best_ppl_score,
        'best_epoch': best_epoch
        })

    save_model(args, model, optimizer)
    wandb.finish()
