import os
import torch

def save_model(args, model, optimizer, scheduler=None):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
    if isinstance(model, torch.nn.DataParallel):
        if scheduler:
            torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': args.current_epoch,}, out)            
        else:
            torch.save({'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': args.current_epoch,}, out)
    else:
        if scheduler:   
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': args.current_epoch,}, out)
        else:           
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': args.current_epoch,}, out)