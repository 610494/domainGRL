import argparse
import functools
import json
import logging
import os
import random
import shutil
from typing import Dict

import dataset.dataset_AES as dataset_lib
import gin
# Import the Method 1 Model
import model.model_ortho as model_lib
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All seeds set to {seed}")

# --- Loss Functions for Method 1 ---
def _orthogonality_loss(z_q, z_d):
    # Compute squared cosine similarity, minimize towards 0
    cosine_sim = F.cosine_similarity(z_q, z_d, dim=1)
    return torch.mean(cosine_sim ** 2)

def _reconstruction_loss(recon, target):
    return F.mse_loss(recon, target)

def _multivariate_gnll_loss(means, targets, covariance, eps=1e-6):
    variance_loss = torch.maximum(torch.logdet(covariance), torch.tensor(eps, device=means.device))
    diff = (means - targets).unsqueeze(-1)
    mean_loss = torch.transpose(diff, 1, 2) @ torch.inverse(covariance) @ diff
    return torch.mean(mean_loss.squeeze() / 2 + variance_loss / 2)

@gin.configurable
class TrainingLoop:
    def __init__(
        self,
        *,
        model: nn.Module,
        save_path: str,
        loss_type: str = 'mgnll',
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        dataset_cls: torch.utils.data.dataset.Dataset = dataset_lib.AESFeatures,
        learning_rate: float = 1e-4,
        batch_size_train: int = 64,
        ssl_layer: int = 11,
        use_wandb: bool = False,
        seed: int = 960,
        # [Fix] Added missing params matching domainGRL.gin
        num_epochs: int = 30,
        weight_decay: float = 0.0,
        covarreg_use: bool = False,    # Placeholder for gin compatibility
        covarreg_lambda: float = 0.1,  # Placeholder for gin compatibility
        # Method 1 Specific Params
        orth_weight: float = 0.1,
        recon_weight: float = 1.0,
        domain_weight: float = 1.0, 
        domain_nums: int = 16, 
    ):
        if not os.path.exists(save_path): os.makedirs(save_path)
        self._save_path = save_path
        logging.basicConfig(filename=os.path.join(save_path, 'train.log'), level=logging.INFO, force=True)

        # Dataset Setup
        dataset_cls_partial = functools.partial(dataset_cls, layer=ssl_layer)
        train_dataset = dataset_cls_partial(dataset_name='audiomos2025-track2-train_list_filter')
        
        # Ensure Domain Config for Model
        gin.bind_parameter('DANNProjectionHead.num_domains', domain_nums)

        # WandB
        self._use_wandb = use_wandb
        if self._use_wandb:
            wandb.init(
                project="aes-ortho", 
                name=f"Ortho_O{orth_weight}_R{recon_weight}_D{domain_nums}_S{seed}", 
                dir=save_path,
                config={
                    "orth_weight": orth_weight,
                    "recon_weight": recon_weight,
                    "seed": seed,
                    "num_epochs": num_epochs
                }
            )

        self._train_loader = dataset_lib.get_dataloader(dataset=train_dataset, batch_size=batch_size_train, num_workers=4, shuffle=True)
        
        def _get_dl(name): 
            return dataset_lib.get_dataloader(dataset_cls_partial(dataset_name=name), batch_size=1, shuffle=False, num_workers=0)
        
        self._valid_loaders = [_get_dl('audiomos2025-track2-dev_list_filter')]
        self._test_loaders = [_get_dl('audiomos2025-track2-eval_list')]

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model(in_shape=train_dataset.features_shape).to(self._device)
        # [Fix] Use weight_decay from args
        self._optimizer = optimizer(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self._loss_type = loss_type
        if loss_type == 'mgnll':
            self._loss_fn = _multivariate_gnll_loss
        else:
            self._loss_fn = nn.MSELoss()
            
        self._domain_loss_fn = nn.CrossEntropyLoss()

        # Weights
        self._orth_weight = orth_weight
        self._recon_weight = recon_weight
        self._domain_weight = domain_weight
        
        # [Fix] Use num_epochs from args
        self._num_epochs = num_epochs
        self._epoch = 0
        self._best_srcc = -1

    def _train_once(self, batch) -> Dict[str, float]:
        features, labels, domain_labels = batch
        features, labels, domain_labels = features.to(self._device), labels.to(self._device), domain_labels.to(self._device)

        # Forward
        outputs = self._model(features)
        
        # Unpack outputs: (main_out, domain_logits, z_q, z_d, recon_output)
        main_out, domain_logits, z_q, z_d, recon_x = outputs
        
        # 1. Main Task Loss
        if self._loss_type == 'mgnll':
            means, covariance = main_out
            loss_main = self._loss_fn(means, labels, covariance)
        else:
            loss_main = self._loss_fn(main_out, labels)

        # 2. Domain Classification Loss (Auxiliary)
        loss_domain = self._domain_loss_fn(domain_logits, domain_labels)

        # 3. Orthogonality Loss
        loss_orth = _orthogonality_loss(z_q, z_d)

        # 4. Reconstruction Loss
        loss_recon = _reconstruction_loss(recon_x, features)

        # Total Loss
        total_loss = loss_main + \
                     (self._domain_weight * loss_domain) + \
                     (self._orth_weight * loss_orth) + \
                     (self._recon_weight * loss_recon)

        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)
        self._optimizer.step()
        
        return {
            'total': total_loss.item(), 
            'main': loss_main.item(),
            'domain': loss_domain.item(), 
            'orth': loss_orth.item(), 
            'recon': loss_recon.item()
        }

    def train(self):
        self._model.train()
        while self._epoch <= self._num_epochs:
            losses = []
            for batch in tqdm.tqdm(self._train_loader, desc=f"Epoch {self._epoch}", ncols=80, unit="step"):
                losses.append(self._train_once(batch))
            
            avg_losses = {k: np.mean([x[k] for x in losses]) for k in losses[0]}
            
            log_msg = f"Epoch {self._epoch} -"
            for name, value in avg_losses.items(): log_msg += f" {name}={value:.4f}"
            logging.info(log_msg)
            
            if self._use_wandb: 
                wandb.log(avg_losses | {"epoch": self._epoch})
            
            self.valid()
            self._epoch += 1

    def _evaluate(self, dataloaders, prefix: str):
        self._model.eval()
        avg_srcc_on_valid_all = []
        label_names = ['Production_Quality', 'Production_Complexity', 'Content_Enjoyment', 'Content_Usefulness']

        for dataloader in dataloaders:
            predictions = {n: [] for n in label_names}
            labels_dict = {n: [] for n in label_names}
            all_srcc = []

            for batch in tqdm.tqdm(dataloader, ncols=80, desc=f"{prefix} on {dataloader.dataset.dataset_name}", unit='step'):
                if len(batch) == 3: feature, label, _ = batch
                else: feature, label = batch
                
                feature = feature.to(self._device)
                with torch.no_grad():
                    # Unpack tuple from model
                    outputs = self._model(feature)
                    main_out = outputs[0]
                    
                    if self._loss_type == 'mgnll': 
                        output_means, _ = main_out
                    else:
                        output_means = main_out
                        
                    output_np = output_means.cpu().numpy()
                    for i, name in enumerate(label_names):
                        predictions[name].extend(output_np[:, i].tolist())
                        labels_dict[name].extend(label[:, i].tolist())

            for name in label_names:
                pred, target = np.array(predictions[name]), np.array(labels_dict[name])
                srcc = scipy.stats.spearmanr(target, pred)[0] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                all_srcc.append(srcc)
                
                if name == 'Production_Quality' and prefix == 'Valid': 
                    avg_srcc_on_valid_all.append(srcc)
                
                logging.info(f"[{dataloader.dataset.dataset_name}][{name}] SRCC={srcc:.4f}")
                if self._use_wandb:
                    wandb.log({f"{prefix}/{dataloader.dataset.dataset_name}_{name}_SRCC": srcc, "epoch": self._epoch})
            
            avg_srcc_all = np.mean(all_srcc)
            logging.info(f"[{dataloader.dataset.dataset_name}_ALL] AVG SRCC={avg_srcc_all:.4f}")

        if prefix == 'Valid' and avg_srcc_on_valid_all:
            current_srcc = np.mean(avg_srcc_on_valid_all)
            if current_srcc > self._best_srcc:
                self._best_srcc = current_srcc
                self.save_model('model_best_state_dict.pt', best_metric=current_srcc)
        self._model.train()

    def valid(self): self._evaluate(self._valid_loaders, 'Valid')
    def test(self):
        state_dict_path = os.path.join(self._save_path, 'model_best_state_dict.pt')
        if os.path.exists(state_dict_path):
            self._model.load_state_dict(torch.load(state_dict_path, map_location=self._device))
            self._evaluate(self._test_loaders, 'Test')

    def save_model(self, model_name='model.pt', best_metric=None):
        torch.save(self._model.state_dict(), os.path.join(self._save_path, model_name))
        with open(os.path.join(self._save_path, "checkpoint.json"), "w") as f:
            json.dump({"epoch": self._epoch, "best_srcc": self._best_srcc}, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_path', type=str, default='configs/domainGRL.gin')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=960)
    # Method 1 Params
    parser.add_argument('--orth_weight', type=float, default=0.1)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--domain_nums', type=int, default=16) 
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    gin.parse_config_file(args.gin_path)
    
    # Bind Params
    gin.bind_parameter('TrainingLoop.orth_weight', args.orth_weight)
    gin.bind_parameter('TrainingLoop.recon_weight', args.recon_weight)
    gin.bind_parameter('TrainingLoop.domain_nums', args.domain_nums)
    gin.bind_parameter('AESFeatures.num_random_domains', args.domain_nums)
    gin.bind_parameter('DANNProjectionHead.num_domains', args.domain_nums)
    gin.bind_parameter('AESFeatures.domain_grouping_strategy', 'k-means') 

    loop = TrainingLoop(
        model=model_lib.DANNProjectionHead, 
        save_path=args.save_path, 
        use_wandb=args.use_wandb, 
        seed=args.seed,
        domain_nums=args.domain_nums
    )
    
    shutil.copyfile(args.gin_path, os.path.join(loop._save_path, 'config.gin'))
    loop.train()
    loop.test()
    if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    main()