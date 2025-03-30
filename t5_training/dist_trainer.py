import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import T5ForConditionalGeneration, AdamW
from transformers.optimization import Adafactor, get_scheduler
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataset_utils import T5Dataset, collator
from tqdm import tqdm
import os


def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def train_t5_on_gpu(
    rank,
    world_size,
    model,
    dataset,
    collate_fn,
    batch_size=8,
    optimizer=None,
    num_epochs=7,
    accumulation_steps=1,
    save_path="t5_finetuned.pt"
):
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
    model = model.to(device)
    
    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    total_steps = len(dataloader) // accumulation_steps * num_epochs
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-3,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
    
    # Create scheduler
    scheduler = get_scheduler(
        "inverse_sqrt",
        optimizer=optimizer,
        num_warmup_steps=10000,
        num_training_steps=total_steps
    )
    
    # Store best model state
    best_loss = float('inf')
    best_model = None
    
    for epoch in range(num_epochs):
        # Important: set the epoch for the sampler
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the start of each epoch
        
        if rank == 0:  # Only show progress on the first GPU
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        else:
            loop = dataloader
            
        for step, batch in enumerate(loop):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough steps
            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Update learning rate scheduler
                scheduler.step()
                
                # Zero gradients after updating weights
                optimizer.zero_grad()
            
            # Track loss for reporting
            total_loss += loss.item() * accumulation_steps
            
            if rank == 0 and isinstance(loop, tqdm):
                # Update progress bar with current loss and learning rate
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                loop.set_postfix(loss=loss.item(), lr=current_lr)
        
        # Make sure all gradients are applied at the end of the epoch
        if (len(dataloader) % accumulation_steps) != 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Calculate and print average loss across all GPUs
        avg_loss = total_loss / len(dataloader)
        loss_tensor = torch.tensor(avg_loss).to(device)
        
        # All-reduce to get the average loss across all processes
        torch.distributed.all_reduce(loss_tensor)
        avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}")
            
            # Save checkpoint (only from the first process to avoid corruptions)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f"{save_path}_epoch{epoch+1}.pt")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict(), f"{save_path}_best.pt")
                print(f"New best model saved with loss: {avg_loss:.4f}")
    
    # Clean up distributed environment
    cleanup()
    
    return model.module if rank == 0 else None

def train_t5_unsupervised(
    model,
    dataset,
    collate_fn,
    batch_size=8,
    num_gpus=2,
    optimizer=None,
    num_epochs=10,
    accumulation_steps=1,
    save_path="t5_finetuned.pt"
):
    """
    Main function to spawn multiple processes for distributed training.
    """
    # Make sure we have GPUs
    assert torch.cuda.is_available(), "No GPU available!"
    world_size = min(torch.cuda.device_count(), num_gpus)
    
    print(f"Starting distributed training on {world_size} GPUs...")
    
    # Spawn processes
    mp.spawn(
        train_t5_on_gpu,
        args=(world_size, model, dataset, collate_fn, batch_size, optimizer, num_epochs, accumulation_steps, save_path),
        nprocs=world_size,
        join=True
    )
    
    print("Training completed!")

def my_collator(batch):
    return collator(batch, tokenizer)

class CollatorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        from dataset_utils import collator
        return collator(batch, self.tokenizer)

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    model_variants = ["google-t5/t5-base", "google-t5/t5-3b", "google/flan-t5-large", "google-t5/t5-11b"]
    device = 'cuda:0'

    model = T5ForConditionalGeneration.from_pretrained(
        model_variants[0],
        torch_dtype=torch.float16,
        token = os.getenv('HF_ACCESS_TOKEN')
    )

    tokenizer = T5Tokenizer.from_pretrained(
        model_variants[0]
    )

    raw_input = '/home/tadesa1/ADBMO-UNLV/data/processed_output_raw.txt'
    with open(raw_input, 'r') as f:
        text = f.readlines()
        text = [line for line in text if line.strip()]


    dataset = T5Dataset(text, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: collator(batch, tokenizer))

    my_collator = CollatorWrapper(tokenizer)
    train_t5_unsupervised(model, dataset, my_collator)
    # train_t5_unsupervised(model, dataset, lambda batch: collator(batch, tokenizer))
