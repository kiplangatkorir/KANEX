import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
import wandb
from .optimizer import KANEXOptimizer

class KANEXTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        self.optimizer = KANEXOptimizer.create_optimizer(
            model,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = KANEXOptimizer.create_scheduler(
            self.optimizer,
            config['num_epochs'],
            config['min_lr']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_dataloader, desc="Training") as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return {'train_loss': total_loss / len(self.train_dataloader)}