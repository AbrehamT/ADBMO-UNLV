import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from functools import partial
from functools import partial
from transformers.models.t5.modeling_t5 import T5Block
from torch.utils.data import DataLoader, DistributedSampler
from transformers import T5ForConditionalGeneration, AdamW, T5Tokenizer
from transformers.optimization import Adafactor, get_scheduler
from transformers.models.t5.modeling_t5 import T5Block
from dataset_utils import T5Dataset, collator
from tqdm import tqdm
import deepspeed
import os
import json
import argparse


def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    # Setting up the addr and port of the coordinating process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    # nccl is the backend that handles distributed computing accross nvidia cuda gpu's
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()


def get_deepspeed_config(offload_optimizer=True, offload_parameters=True, stage=3):
    """
    Create DeepSpeed configuration
    """
    config = {
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            } if offload_optimizer else None,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            } if offload_parameters else None,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "sub_group_size": 1e8,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": 10000,
                "total_num_steps": "auto"
            }
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None
        },
        "wall_clock_breakdown": True
    }
    
    return config


def train_t5_with_deepspeed_fsdp(
    rank,
    world_size,
    model,
    dataset,
    collate_fn,
    batch_size=1,
    use_deepspeed=True,
    use_fsdp=False,
    num_epochs=7,
    accumulation_steps=1,
    save_path="t5_finetuned.pt"
):
    """
    Train T5 with DeepSpeed and/or FSDP
    """
    # Setup the distributed environment
    setup(rank, world_size)
    
    # Create DistributedSampler to handle data distribution
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Create dataloader with the sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Move model to the correct device
    device = torch.device(f"cuda:{rank}")
    
    # Calculate total steps for scheduler
    total_steps = len(dataloader) // accumulation_steps * num_epochs
    
    if use_deepspeed:
        # Configure DeepSpeed
        ds_config = get_deepspeed_config()
        
        # Set dynamic configurations
        ds_config['train_batch_size'] = batch_size * world_size
        ds_config['gradient_accumulation_steps'] = accumulation_steps
        ds_config['scheduler']['params']['total_num_steps'] = total_steps
        ds_config['scheduler']['params']['warmup_max_lr'] = 1e-5
        
        # Write config to file
        with open('ds_config.json', 'w') as f:
            json.dump(ds_config, f)
        
        # Initialize DeepSpeed
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config='ds_config.json',
            dist_init_required=False  # Already initialized
        )
        
    elif use_fsdp:
        # Move model to device first
        model = model.to(device)
        
        # Define mixed precision policy
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            # Keep buffers in full precision
            buffer_dtype=torch.float32,
            # Perform reduction in fp16
            reduce_dtype=torch.float16
        )
        
        # CPU Offload configuration
        cpu_offload = CPUOffload(offload_params=True)
        
        # Use the simplest FSDP setup without custom wrapping policies
        # as those are causing compatibility issues
        try:
            print("Initializing FSDP model without auto-wrapping...")
            # This uses the simplest FSDP without auto-wrapping which should
            # be more compatible across PyTorch versions
            model = FSDP(
                model,
                mixed_precision=mixed_precision_policy,
                cpu_offload=cpu_offload,
                device_id=rank
            )
            print("FSDP initialized successfully!")
        except Exception as e:
            print(f"Error initializing FSDP: {e}")
            print("Falling back to DDP")
            
            # Fall back to basic DDP (no FSDP)
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[rank], output_device=rank)
            use_fsdp = False  # Mark FSDP as failed for later code
        
        # Create optimizer
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-5,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
        
        # Create scheduler
        lr_scheduler = get_scheduler(
            "inverse_sqrt",
            optimizer=optimizer,
            num_warmup_steps=10000,
            num_training_steps=total_steps
        )
    else:
        # Move model to the correct device
        model = model.to(device)
        
        # Use DDP (as a fallback)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Create optimizer
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-5,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
        
        # Create scheduler
        lr_scheduler = get_scheduler(
            "inverse_sqrt",
            optimizer=optimizer,
            num_warmup_steps=10000,
            num_training_steps=total_steps
        )
    
    # Store best model state
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Important: set the epoch for the sampler
        sampler.set_epoch(epoch)
        
        # Set model to train mode
        model.train()
        total_loss = 0.0
        
        # Only show progress on the first GPU
        if rank == 0:
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        else:
            loop = dataloader
            
        for step, batch in enumerate(loop):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if use_deepspeed:
                # Forward pass with DeepSpeed
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Backward pass with DeepSpeed
                model.backward(loss)
                
                # Update weights if we've accumulated enough steps
                if (step + 1) % accumulation_steps == 0:
                    model.step()
                    
            else:
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if we've accumulated enough steps
                if (step + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    if not use_fsdp:  # FSDP handles clipping internally
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    
                    # Update learning rate scheduler
                    lr_scheduler.step()
                    
                    # Zero gradients after updating weights
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
            
            # Track loss for reporting
            total_loss += loss.item() * accumulation_steps
            
            if rank == 0 and isinstance(loop, tqdm):
                # Update progress bar with current loss
                if use_deepspeed:
                    current_lr = model.get_lr()[0]
                else:
                    current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                
                loop.set_postfix(loss=loss.item(), lr=current_lr)
        
        # Make sure all gradients are applied at the end of the epoch
        if not use_deepspeed and (len(dataloader) % accumulation_steps) != 0:
            # Gradient clipping
            if not use_fsdp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Calculate and print average loss across all GPUs
        avg_loss = total_loss / len(dataloader)
        loss_tensor = torch.tensor(avg_loss).to(device)
        
        # All-reduce to get the average loss across all processes
        torch.distributed.all_reduce(loss_tensor)
        avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            
            # Save checkpoint (only from the first process to avoid corruptions)
            if use_deepspeed:
                # DeepSpeed checkpoint
                model.save_checkpoint(save_dir=f"{save_path}_epoch{epoch+1}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    model.save_checkpoint(save_dir=f"{save_path}_best")
                    print(f"New best model saved with loss: {avg_loss:.4f}")
                
            elif use_fsdp:
            # FSDP checkpoint saving
                try:
                    from torch.distributed.fsdp import FullStateDictConfig
                    from torch.distributed.fsdp import StateDictType
                    
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                        model_state = model.state_dict()
                        if rank == 0:
                            checkpoint = {
                                'epoch': epoch + 1,
                                'model_state_dict': model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': avg_loss,
                            }
                            torch.save(checkpoint, f"{save_path}_epoch{epoch+1}.pt")
                            
                            if avg_loss < best_loss:
                                best_loss = avg_loss
                                torch.save(model_state, f"{save_path}_best.pt")
                                print(f"New best model saved with loss: {avg_loss:.4f}")
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Advanced FSDP checkpointing failed: {e}")
                    print("Falling back to basic checkpointing")
                    
                    # Basic checkpointing without state_dict_type
                    if rank == 0:
                        try:
                            checkpoint = {
                                'epoch': epoch + 1,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': avg_loss,
                            }
                            torch.save(checkpoint, f"{save_path}_epoch{epoch+1}.pt")
                            
                            if avg_loss < best_loss:
                                best_loss = avg_loss
                                torch.save(model.state_dict(), f"{save_path}_best.pt")
                                print(f"New best model saved with loss: {avg_loss:.4f}")
                        except Exception as e2:
                            print(f"Error saving checkpoint: {e2}")
                            print("Cannot save FSDP model checkpoint")
            else:
                # Regular DDP checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, f"{save_path}_epoch{epoch+1}.pt")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.module.state_dict(), f"{save_path}_best.pt")
                    print(f"New best model saved with loss: {avg_loss:.4f}")
    
    # Clean up distributed environment
    cleanup()


def train_t5_distributed(
    model,
    dataset,
    collate_fn,
    batch_size=1,
    num_gpus=2,
    training_method="deepspeed",  # "deepspeed", "fsdp", or "ddp"
    num_epochs=10,
    accumulation_steps=1,
    save_path="t5_finetuned"
):
    """
    Main function to spawn multiple processes for distributed training.
    """
    # Make sure we have GPUs
    import torch
    assert torch.cuda.is_available(), "No GPU available!"
    world_size = min(torch.cuda.device_count(), num_gpus)
    
    print(f"Starting distributed training on {world_size} GPUs using {training_method}...")
    
    # Safer default - if FSDP has issues, fall back to DeepSpeed
    if training_method.lower() == "fsdp":
        try:
            import torch.distributed.fsdp
            print("FSDP is available. Will use FSDP.")
            # Check the PyTorch version
            torch_version = torch.__version__.split('.')
            major, minor = int(torch_version[0]), int(torch_version[1])
            
            # If using older PyTorch versions, recommend DeepSpeed instead
            if major < 2 and minor < 0:
                print(f"Warning: PyTorch {torch.__version__} may have FSDP compatibility issues.")
                print("Consider using DeepSpeed for better compatibility.")
                
        except (ImportError, AttributeError):
            print("WARNING: FSDP not properly available. Falling back to DeepSpeed.")
            training_method = "deepspeed"
    
    # Set use_deepspeed and use_fsdp based on training_method
    use_deepspeed = (training_method.lower() == "deepspeed")
    use_fsdp = (training_method.lower() == "fsdp")
    
    # Spawn processes
    mp.spawn(
        train_t5_with_deepspeed_fsdp,
        args=(world_size, model, dataset, collate_fn, batch_size, use_deepspeed, use_fsdp, num_epochs, accumulation_steps, save_path),
        nprocs=world_size,
        join=True
    )
    
    print("Training completed!")


class CollatorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        from dataset_utils import collator
        return collator(batch, self.tokenizer)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train T5 with DeepSpeed, FSDP, or DDP')
    parser.add_argument('--model_name', type=str, default="google-t5/t5-base", 
                        choices=["google-t5/t5-base", "google-t5/t5-3b", "google/flan-t5-large", "google-t5/t5-11b", "google/flan-t5-xxl"],
                        help='T5 model variant to use')
    parser.add_argument('--training_method', type=str, default="deepspeed", 
                        choices=["deepspeed", "fsdp", "ddp"],
                        help='Training method (deepspeed, fsdp, or ddp)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--input_file', type=str, default='/home/tadesa1/ADBMO-UNLV/data/processed_output_raw.txt',
                        help='Input text file for training')
    parser.add_argument('--save_path', type=str, default='t5_finetuned',
                        help='Path to save model checkpoints')
                        
    args = parser.parse_args()
    
    # Suppress TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Print PyTorch version for debugging
    print(f"PyTorch version: {torch.__version__}")
    
    # Set NCCL_DEBUG for better CUDA debugging
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Initialize model
    print(f"Loading model: {args.model_name}")
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        token=os.getenv('HF_ACCESS_TOKEN')
    )
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    
    # Load dataset
    print(f"Loading dataset from: {args.input_file}")
    with open(args.input_file, 'r') as f:
        text = f.readlines()
        text = [line for line in text if line.strip()]
    
    # Create dataset
    dataset = T5Dataset(text, tokenizer)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Create collator
    my_collator = CollatorWrapper(tokenizer)
    
    # Start training
    train_t5_distributed(
        model=model,
        dataset=dataset,
        collate_fn=my_collator,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        training_method=args.training_method,
        num_epochs=args.num_epochs,
        accumulation_steps=args.accumulation_steps,
        save_path=args.save_path
    )