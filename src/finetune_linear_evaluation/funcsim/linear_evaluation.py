import os
import sys
import h5py
import time
import wandb
import torch
import argparse
import numpy as np
import torch.nn as nn
from funcsimcls import funcsimcls
from torch.nn import CosineEmbeddingLoss
from binarycorpdataset import binarycorpdataset
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0,"src/utils")

from yaml_config_hook import yaml_config_hook
from save_model import save_model

def train(args, loader, model, criterion, optimizer):
    loss_epoch = 0
    model.train()
    for step, (seq1, seq2) in enumerate(loader):
        optimizer.zero_grad()

        seq1 = {feature: value.cuda(non_blocking=True) for feature, value in seq1.items()}
        seq2 = {feature: value.cuda(non_blocking=True) for feature, value in seq2.items()}

        anchor = model(seq1)
        pos = model(seq2)
        pos_target = torch.ones(args.logistic_batch_size)
        loss = criterion(anchor, pos, pos_target.cuda())
        neg = torch.roll(pos, shifts=-1, dims=0)
        neg_target = torch.ones(args.logistic_batch_size) * -1
        loss += criterion(anchor, neg, neg_target.cuda())

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        if step % args.show_step == 0:
            print(
                 f"Train Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t"
             )

    return loss_epoch


def test(args, loader, model, num_testdata):
    hdf5_pt = os.path.join(args.model_path, 'embeddings.hdf5')
    model.eval()
    total_inference_time = 0
    total_samples = 0

    with h5py.File(hdf5_pt, 'w') as f:
        O0_embedding_dataset = f.create_dataset('O0_embedding_dataset', (num_testdata, args.projection_dim), dtype='f')
        O1_embedding_dataset = f.create_dataset('O1_embedding_dataset', (num_testdata, args.projection_dim), dtype='f')
        O2_embedding_dataset = f.create_dataset('O2_embedding_dataset', (num_testdata, args.projection_dim), dtype='f')
        O3_embedding_dataset = f.create_dataset('O3_embedding_dataset', (num_testdata, args.projection_dim), dtype='f')
        Os_embedding_dataset = f.create_dataset('Os_embedding_dataset', (num_testdata, args.projection_dim), dtype='f')

        O0_embedding_mask = f.create_dataset('O0_embedding_mask', (num_testdata,), dtype='bool')
        O1_embedding_mask = f.create_dataset('O1_embedding_mask', (num_testdata,), dtype='bool')
        O2_embedding_mask = f.create_dataset('O2_embedding_mask', (num_testdata,), dtype='bool')
        O3_embedding_mask = f.create_dataset('O3_embedding_mask', (num_testdata,), dtype='bool')
        Os_embedding_mask = f.create_dataset('Os_embedding_mask', (num_testdata,), dtype='bool')

        for step, (seq1, seq2, seq3, seq4, seq5) in enumerate(loader):
            model.zero_grad()

            total_samples += seq1['input_ids'].shape[0]

            with torch.no_grad():
                start_time = time.time()

                start_index = step * seq1['input_ids'].shape[0]
                end_index = start_index + seq1['input_ids'].shape[0]

                O0_embedding_mask[start_index:end_index] = [opt != 'Fake' for opt in seq1['opti']]
                O1_embedding_mask[start_index:end_index] = [opt != 'Fake' for opt in seq2['opti']]
                O2_embedding_mask[start_index:end_index] = [opt != 'Fake' for opt in seq3['opti']]
                O3_embedding_mask[start_index:end_index] = [opt != 'Fake' for opt in seq4['opti']]
                Os_embedding_mask[start_index:end_index] = [opt != 'Fake' for opt in seq5['opti']]

                seq1 = {feature: value.cuda(non_blocking=True) for feature, value in seq1.items() if feature != 'opti'}
                seq2 = {feature: value.cuda(non_blocking=True) for feature, value in seq2.items() if feature != 'opti'}
                seq3 = {feature: value.cuda(non_blocking=True) for feature, value in seq3.items() if feature != 'opti'}
                seq4 = {feature: value.cuda(non_blocking=True) for feature, value in seq4.items() if feature != 'opti'}
                seq5 = {feature: value.cuda(non_blocking=True) for feature, value in seq5.items() if feature != 'opti'}
                
                O0_embedding_dataset[start_index:end_index] = model(seq1).detach().cpu().numpy()
                O1_embedding_dataset[start_index:end_index] = model(seq2).detach().cpu().numpy()
                O2_embedding_dataset[start_index:end_index] = model(seq3).detach().cpu().numpy()
                O3_embedding_dataset[start_index:end_index] = model(seq4).detach().cpu().numpy()
                Os_embedding_dataset[start_index:end_index] = model(seq5).detach().cpu().numpy()

                if step % args.show_step == 0:
                    print(
                        f"Test Step [{step}/{len(loader)}]\t"
                    )

                end_time = time.time()
                total_inference_time += (end_time - start_time)

    k = 1
    pool_sizes = [2, 10, 32, 128, 512, 1000, 10000]
    opti_pairs = [('O0','O3'), ('O1','O3'), ('O2','O3'),('O0','Os'), ('O1','Os'), ('O2','Os')]
    metric = {}

    with h5py.File(hdf5_pt, 'r') as f:
        for pool_size in pool_sizes:
            for (opt1, opt2) in opti_pairs:
                start_time = time.time()

                ebd1, ebd2 = f[f"{opt1}_embedding_dataset"][:], f[f"{opt2}_embedding_dataset"][:]
                mask1, mask2 = f[f"{opt1}_embedding_mask"][:], f[f"{opt2}_embedding_mask"][:]

                real_indices = np.where(mask1 & mask2)[0]

                mrr_sum = 0
                recall_at_k_count = 0

                for i in range(0, len(real_indices), pool_size):
                    chunk_indices = real_indices[i:i + pool_size]
                    if len(chunk_indices) != pool_size:
                        continue
                    
                    ebd1_chunk = ebd1[chunk_indices]
                    ebd2_chunk = ebd2[chunk_indices]
                    
                    cosine_sim_matrix = cosine_similarity(ebd1_chunk, ebd2_chunk)

                    for j in range(pool_size):
                        similarities = cosine_sim_matrix[j]
                        sorted_index = np.argsort(-similarities)
                        rank = np.where(sorted_index == j)[0][0] + 1
                        if rank <= k:
                            recall_at_k_count += 1
                        mrr_sum += 1 / rank

                total_valid_examples = (len(real_indices) // pool_size) * pool_size
                mrr_sum /= total_valid_examples
                recall_at_k_count /= total_valid_examples

                metric[str(pool_size), opt1, opt2, 'mrr'] = mrr_sum
                metric[str(pool_size), opt1, opt2, 'recall'] = recall_at_k_count

                end_time = time.time()
                total_inference_time += (end_time - start_time)

    average_inference_time_per_sample = total_inference_time / total_samples

    return metric, pool_sizes, opti_pairs, average_inference_time_per_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("src/0_environment_setup/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.margin = 0.5

    data_size = None
    if ('95_5' in args.model_path) or ('6_reload_then_finetune' in args.model_path):
        assert args.model_path.split('/')[-1] != '95_5', 'need data size'
        data_size, dataset_split, model_size  = args.model_path.split('/')[-1], args.model_path.split('/')[-2], args.model_path.split('/')[-3]
    else:
        assert 'per' not in args.model_path.split('/')[-1], 'no need data size'
        dataset_split, model_size  = args.model_path.split('/')[-1], args.model_path.split('/')[-2]

    if args.dataset_arch == 'x86':
        args.dataset_pt = f'workdir/4_prepare_finetune_dataset/x64/funcsim/{dataset_split}'
        projectname = f"GBME_funcsim_{dataset_split}_{args.backbone.lower()}_{model_size}_x86"
    elif args.dataset_arch == 'all':
        args.dataset_pt = f'workdir/4_prepare_finetune_dataset/all_architecture/funcsim/{dataset_split}'
        projectname = f"GBME_funcsim_{dataset_split}_{args.backbone.lower()}_{model_size}_allarch"
    else:
        raise Exception('wrong dataset')

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
            pass
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    finetune_from = 'funcsig'
    pretrain_split = '95_5'

    name_list = ['funcsim', args.backbone.lower()]
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
    wandb.init(project=projectname, entity='lhxxh',  name='_'.join(name_list))
    wandb.config.batch_size = args.logistic_batch_size
    wandb.config.learning_rate = 1e-5

    print('Supervised finetune arguments')
    for k, v in vars(args).items():
        print(f'{k} : {v}')

    train_dataset = binarycorpdataset(args, 'train')
    test_dataset = binarycorpdataset(args, 'test')

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
    assert 'funcsim' in args.model_path, 'dataset and path not compatible'
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

    model = funcsimcls(args)
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
            # For malconv
            checkpoint_state_dict.pop('out_proj.weight', None)
            checkpoint_state_dict.pop('out_proj.bias', None)
            # For roberta
            checkpoint_state_dict.pop('model.classifier.out_proj.weight', None)
            checkpoint_state_dict.pop('model.classifier.out_proj.bias', None)
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
    criterion = CosineEmbeddingLoss(margin=args.margin)

    print('Start finetuning ...')
    args.current_epoch = args.logistic_start_epoch 

    # Early stopping parameters
    best_mrr_32 = 0.0
    best_mrr_10000 = 0.0
    best_epoch = 0
    early_stopping_counter = 0
    early_stopping_patience = args.early_stopping_patience


    for epoch in range(args.logistic_start_epoch, args.logistic_epochs):
        loss_epoch = train(
            args, train_loader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t"
        )
        args.current_epoch += 1

        if epoch % args.save_epoch == 0:
            save_model(args, model, optimizer)

        if epoch % args.validation_epoch == 0:
            print("Start Validation ...")
            metric, pool_sizes, opti_pairs, average_inference_time = test(
                args, test_loader, model, len(test_dataset)
            )
            
            for key, value in metric.items():
                print(f"[Validation current configuration {key}] : {value}") 

            pool_size_mrr_record ={}
            for pool_size in pool_sizes:
                mrr_averaged = 0
                recall_averaged = 0
                for (opt1, opt2) in opti_pairs:
                    mrr_averaged += metric[str(pool_size), opt1, opt2, 'mrr']
                    recall_averaged += metric[str(pool_size), opt1, opt2, 'recall']

                print(f'[Average] Pool size : {pool_size} MRR : {mrr_averaged/len(opti_pairs)} Recall : {recall_averaged/len(opti_pairs)}')
                pool_size_mrr_record[str(pool_size)] = mrr_averaged/len(opti_pairs)

            # Early stopping check
            if pool_size_mrr_record['32'] > best_mrr_32 or pool_size_mrr_record['10000'] > best_mrr_10000:
                best_mrr_32 = pool_size_mrr_record['32']
                best_mrr_10000 = pool_size_mrr_record['10000']
                best_epoch = args.current_epoch
                early_stopping_counter = 0 
            else:
                early_stopping_counter += 1

            wandb.log({
                'epoch': args.current_epoch,
                'MRR_32': pool_size_mrr_record['32'],
                'MRR_10000': pool_size_mrr_record['10000'],
                'inference_time': average_inference_time,
                })

            # Check if early stopping is triggered
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered. Stopping training at epoch {args.current_epoch}.")
                break 

    wandb.log({
        'best_mrr_32': best_mrr_32,
        'best_mrr_10000': best_mrr_10000,
        'best_f1_epoch': best_epoch
        })

    save_model(args, model, optimizer)
    wandb.finish()



