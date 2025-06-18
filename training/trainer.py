import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from src.models.clip_model import CLIPModel
from transformers import get_cosine_schedule_with_warmup

class CLIPTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Init dataloaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Init optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        # Init lr scheduler
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=config.max_epochs,
        #     eta_min=self.config.min_lr,
        # )

        # 计算总步数：epoch 数 × 每 epoch 的 batch 数
        total_steps = config.max_epochs * len(self.train_loader)
        # Cosine + linear warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # 半个周期常用
            last_epoch=-1
        )

        # training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # saving and logging directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # open mixed precision training
        # self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision)

        # init tensorboard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            # texts = batch['text'].to(self.device)

            # use the tokenizer to get input_ids and attention_mask
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward with mixed precision
            # with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                # outputs = self.model(images, texts, return_loss=True)
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True
                )
                loss = outputs["loss"]

            # backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # record log
            if self.global_step % self.config.log_freq == 0:
                self._log_training_sep(loss.item(), outputs)

        return total_loss / num_batches

    def _log_training_sep(self, loss, outputs):
        """use tensorboard to log information"""
        self.writer.add_scalar('Loss/train', loss, self.global_step)

        if 'image2text_loss' in outputs and 'text2image_loss' in outputs:
            # self.writer.add_scalar('Loss/image_loss', outputs['image2text_loss'].item(), self.global_step)
            self.writer.add_scalar('Loss/image_loss', outputs['image2text_loss'], self.global_step)
            # self.writer.add_scalar('Loss/text_loss', outputs['text2image_loss'].item(), self.global_step)
            self.writer.add_scalar('Loss/text_loss', outputs['text2image_loss'], self.global_step)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, self.global_step)

        if hasattr(self.model, 'logit_scale'):
            temp = self.model.logit_scale.exp().item()
            self.writer.add_scalar('Params/temperature', temp, self.global_step)

        # The following code will add calculation
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         self.writer.add_histogram(f'Gradients/{name}', param.grad, self.global_step)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                # texts = batch['text'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    # outputs = self.model(images, texts, return_loss=True)
                    outputs = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_loss=True
                    )
                    loss = outputs["loss"]
                total_loss += loss.item()
            avg_loss = total_loss / num_batches
            return avg_loss
    def train(self):
        print(f"Starting training for {self.config.max_epochs} epochs...")

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch

            # only train for one epoch
            train_loss = self.train_epoch()

            if epoch % self.config.eval_freq == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                #
                # Log validation loss to TensorBoard
                self.writer.add_scalar('Loss/validation', val_loss, self.global_step)
                self.writer.add_scalars('Loss/Comparison', {'train': train_loss, 'validation': val_loss},
                                        self.global_step)
                # save the best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

            if epoch % self.config.save_freq == 0:
                self.save_checkpoint()

            # update lr
            if self.scheduler:
                self.scheduler.step()

        self.writer.close()
        print("Training complete")
        print("Check the output of TensorBoard, use: tensorboard --logdir ./training/test_logs")

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # save normal checkpoint
        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        # save best model
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loading checkpoint from {checkpoint_path}, epoch: {self.current_epoch}")


if __name__ == '__main__':
    print("--- Running Local Smoke Test ---")


    # --- 步骤 1: 创建模拟配置 ---
    class MockConfig:
        # 训练器参数
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 4
        num_workers = 0
        max_epochs = 2
        learning_rate = 1e-4
        weight_decay = 1e-2
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        min_lr = 1e-6
        mixed_precision = torch.cuda.is_available()
        log_freq = 5
        eval_freq = 1
        save_freq = 1
        save_dir = "./test_checkpoints_real_model"
        log_dir = "./test_logs_real_model"

        # CLIPModel 参数
        vision_encoder = 'resnet50'  # 或 'vit'
        embed_dim = 512
        temperature = 0.07
        dropout = 0.1

        # Vision Encoder (ViT) 参数
        image_size = 224
        patch_size = 32
        vit_depth = 12
        vit_num_heads = 8
        vit_mlp_ratio = 4.0

        # Text Encoder 参数
        vocab_size = 49408  # CLIP 的标准 vocab size
        max_text_length = 77
        text_heads = 8
        text_layers = 12


    # --- 步骤 2: 创建模拟CLIP模型 ---
    class MockCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            # 模仿logit_scale，以便Trainer可以记录它
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
            # 一个简单的线性层来处理输入
            self.dummy_param = nn.Linear(10, 1)

        def forward(self, images, texts, return_loss=False):
            # 模拟模型处理过程，输出一个随机损失
            dummy_output = self.dummy_param(torch.randn(images.size(0), 10, device=images.device))
            loss = torch.mean(torch.abs(dummy_output))  # 一个简单的损失函数

            # 返回Trainer期望的字典结构
            return {
                "loss": loss,
                "image2text_loss": loss / 2,  # 模拟分项损失
                "text2image_loss": loss / 2,  # 模拟分项损失
            }


    # --- 步骤 3: 创建模拟数据集 ---
    class MockDataset(Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 224, 224),
                "text": torch.randint(0, MockConfig.vocab_size, (MockConfig.max_text_length,)),
            }


    # --- 步骤 4: 实例化并运行 ---
    # 在开始前清理旧的测试目录
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")

    # 实例化所有组件
    config = MockConfig()
    model = CLIPModel(config)
    train_dataset = MockDataset(size=100)
    val_dataset = MockDataset(size=20)

    print(f"Test configuration: running on '{config.device}' for {config.max_epochs} epochs.")

    # 创建并启动Trainer
    trainer = CLIPTrainer(model, train_dataset, val_dataset, config)
    trainer.train()

    print("\n--- Local Smoke Test Finished Successfully! ---")
    print(f"Checkpoints saved in: {config.save_dir}")
    print(f"TensorBoard logs saved in: {config.log_dir}")
    print("To view logs, run: tensorboard --logdir ./test_logs")
