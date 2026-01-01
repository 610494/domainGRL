import argparse
import functools
import json
import logging
import os
import random
import shutil
from typing import Dict, Sequence

# [修改] 使用 AES dataset
import dataset.dataset_AES as dataset_lib
import gin
import model.model_domainGRL as model_lib
import numpy as np
import scipy
import torch
import torch.nn as nn
import tqdm
import wandb


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All seeds set to {seed}")

def _multivariate_gnll_loss(means, targets, covariance, eps=1e-6):
    variance_loss = torch.maximum(torch.logdet(covariance), torch.tensor(eps, device=means.device))
    diff = (means - targets).unsqueeze(-1)
    mean_loss = torch.transpose(diff, 1, 2) @ torch.inverse(covariance) @ diff
    return torch.mean(mean_loss.squeeze() / 2 + variance_loss / 2)

def _correlation_regularization_loss(covariance):
    """Old naive regularization: penalize negative correlations for MOS."""
    if covariance.shape[1] > 1:
        mos_covariances = covariance[:, 0, 1:] 
        penalty = torch.relu(-mos_covariances)
        return torch.mean(penalty)
    return torch.tensor(0.0, device=covariance.device)

def _prior_knowledge_loss(predicted_cov, prior_corr_matrix):
    """
    計算預測的 Covariance Matrix 與先驗 Correlation Matrix 的結構差異 (MSE / Frobenius Norm)。
    """
    # 1. 將 Covariance 轉換為 Correlation Matrix
    # 取得對角線元素 (Variance) -> (Batch, N)
    variances = torch.diagonal(predicted_cov, dim1=-2, dim2=-1)
    
    # 取得標準差 (Std) -> (Batch, N)
    stds = torch.sqrt(variances + 1e-8) 
    
    # 計算外積矩陣 (Batch, N, N)，每個元素 (i, j) 為 std[i] * std[j]
    denominator = torch.bmm(stds.unsqueeze(2), stds.unsqueeze(1))
    
    # 預測的相關係數矩陣
    predicted_corr = predicted_cov / (denominator + 1e-8)
    
    # 2. 計算與 Prior 的 MSE Loss
    target = prior_corr_matrix.to(predicted_cov.device)
    
    # 計算差異平方
    diff = predicted_corr - target
    # loss = torch.mean(diff ** 2)
    loss = torch.mean(torch.abs(diff))
    
    return loss

@gin.configurable
class TrainingLoop:
    def __init__(
        self,
        *,
        model: nn.Module,
        save_path: str,
        loss_type: str = 'mgnll',
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        weight_decay: float = 0.0,
        dataset_cls: torch.utils.data.dataset.Dataset = dataset_lib.AESFeatures,
        num_epochs: int = 500,
        learning_rate: float = 1e-4,
        batch_size_train: int = 64,
        ssl_layer: int = 11,
        use_wandb: bool = False,
        seed: int = 960,
        covarreg_use: bool = False,
        covarreg_lambda: float = 0.1,
        prior_corr_matrix: Sequence[Sequence[float]] | None = None, # [新增] 接收 Prior Matrix
        domain_weight: float = 1.0,
        domain_grouping_strategy: str | None = None,
        domain_nums: int | None = None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self._save_path = save_path
        log_path = os.path.join(save_path, 'train.log')
        logging.basicConfig(filename=log_path, level=logging.INFO, force=True)

        # Dataset setup
        dataset_cls_partial = functools.partial(dataset_cls, layer=ssl_layer)
        train_dataset = dataset_cls_partial(dataset_name='audiomos2025-track2-train_list_filter')
        
        # [修改] 自動偵測 Domain 數量 (針對 GT 策略)
        actual_domain_nums = domain_nums 
        if hasattr(train_dataset, 'num_domains'):
            detected_domains = train_dataset.num_domains
            logging.info(f"[Auto-Detect] Dataset reported {detected_domains} domains.")
            
            if domain_grouping_strategy == 'GT' or actual_domain_nums is None:
                actual_domain_nums = detected_domains
                gin.bind_parameter('DANNProjectionHead.num_domains', actual_domain_nums)
                print(f"[Gin Override] DANNProjectionHead.num_domains set to {actual_domain_nums}")

        # Initialize WandB
        self._use_wandb = use_wandb
        if self._use_wandb:
            run_name = f"AES_DW{domain_weight}_DS{domain_grouping_strategy}_DN{actual_domain_nums}_CL{covarreg_lambda}_L1_S{seed}"
            wandb.init(
                project="aes-training",
                name=run_name,
                config={
                    "loss_type": loss_type,
                    "optimizer": optimizer.__name__,
                    "lr": learning_rate,
                    "batch_size": batch_size_train,
                    "seed": seed,
                    "covarreg_lambda": covarreg_lambda,
                    "domain_weight": domain_weight,
                    "domain_strategy": domain_grouping_strategy,
                    "domain_nums": actual_domain_nums,
                    "use_prior_loss": (prior_corr_matrix is not None)
                },
                dir=save_path
            )

        self._train_loader = dataset_lib.get_dataloader(
            dataset=train_dataset, batch_size=batch_size_train, num_workers=4, shuffle=True
        )
        
        def _get_dataloaders(names):
            return [dataset_lib.get_dataloader(
                    dataset=dataset_cls_partial(dataset_name=name), batch_size=1, shuffle=False, num_workers=0
                ) for name in names]
        
        self._valid_loaders = _get_dataloaders(['audiomos2025-track2-dev_list_filter'])
        self._test_loaders = _get_dataloaders(['audiomos2025-track2-eval_list'])

        # Model setup
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model(in_shape=train_dataset.features_shape).to(self._device)
        self._optimizer = optimizer(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._optimizer.zero_grad()

        self._loss_type = loss_type
        self._domain_weight = domain_weight
        self._domain_loss_fn = nn.CrossEntropyLoss()
        if loss_type == 'mgnll':
            self._loss_fn = _multivariate_gnll_loss
        elif loss_type == 'mse':
            self._loss_fn = nn.MSELoss()
        else:
            raise ValueError(f'{loss_type=} is invalid.')

        self._covarreg_use = covarreg_use
        self._covarreg_lambda = covarreg_lambda
        
        # [處理 Prior Matrix]
        if prior_corr_matrix is not None:
            logging.info("[Prior] Loading Prior Correlation Matrix...")
            self._prior_matrix = torch.tensor(prior_corr_matrix, dtype=torch.float32)
            assert self._prior_matrix.shape[0] == self._prior_matrix.shape[1], "Prior matrix must be square"
            self._use_prior_loss = True
            logging.info(f"[Prior] Matrix:\n{self._prior_matrix}")
        else:
            self._use_prior_loss = False
            self._prior_matrix = None

        self._num_epochs = num_epochs
        self._epoch = 0
        self._best_srcc = -1
        self._all_losses = []

    def _train_once(self, batch) -> Dict[str, float]:
        features, labels, domain_labels = batch
        features, labels, domain_labels = features.to(self._device), labels.to(self._device), domain_labels.to(self._device)

        loss_corr = torch.tensor(0.0, device=self._device)
        if self._loss_type == 'mgnll':
            (means, covariance), domain_logits = self._model(features, domain_weight=self._domain_weight)
            loss_main = self._loss_fn(means, labels, covariance) 
            
            # [修改 Logic] 根據設定決定使用哪種 Regularization
            if self._covarreg_use:
                if self._use_prior_loss:
                    loss_corr = _prior_knowledge_loss(covariance, self._prior_matrix)
                else:
                    loss_corr = _correlation_regularization_loss(covariance)
        else: 
            means, domain_logits = self._model(features, domain_weight=self._domain_weight)
            loss_main = self._loss_fn(means, labels) 
        
        loss_domain = self._domain_loss_fn(domain_logits, domain_labels)
        total_loss = loss_main + self._covarreg_lambda * loss_corr + self._domain_weight * loss_domain

        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)
        self._optimizer.step()
        self._optimizer.zero_grad()
        
        return { 
            'total_loss': total_loss.item(), 'main_loss': loss_main.item(), 
            'corr_loss': loss_corr.item(), 'domain_loss': loss_domain.item()
        }

    def train(self, valid_each_epoch: bool = True):
        self._model.train()
        while self._epoch <= self._num_epochs:
            self._all_losses = []
            for batch in tqdm.tqdm(self._train_loader, ncols=0, desc=f"Epoch {self._epoch}", unit="step"):
                self._all_losses.append(self._train_once(batch))

            if self._all_losses:
                avg_losses = {k: np.mean([d[k] for d in self._all_losses]) for k in self._all_losses[0]}
            else:
                avg_losses = {'total_loss': 0.0}

            log_msg = f"Epoch {self._epoch} -"
            for name, value in avg_losses.items(): log_msg += f" Avg {name}={value:.4f}"
            logging.info(log_msg)
            
            if self._use_wandb:
                wandb.log({f"train/{k}": v for k, v in avg_losses.items()} | {"epoch": self._epoch})

            if valid_each_epoch: self.valid()
            self._epoch += 1

    def _evaluate(self, dataloaders, prefix: str):
        self._model.eval()
        avg_srcc_on_valid_all = []
        label_names = ['Production_Quality', 'Production_Complexity', 'Content_Enjoyment', 'Content_Usefulness']

        for dataloader in dataloaders:
            predictions = {n: [] for n in label_names}
            labels = {n: [] for n in label_names}
            all_mse, all_pcc, all_srcc = [], [], []

            for batch in tqdm.tqdm(dataloader, ncols=0, desc=f"{prefix} on {dataloader.dataset.dataset_name}", unit='step'):
                if len(batch) == 3: feature, label, _ = batch
                else: feature, label = batch
                
                feature = feature.to(self._device)
                with torch.no_grad():
                    output, _ = self._model(feature)
                    if self._loss_type == 'mgnll': output, _ = output
                    output = output.cpu().numpy()
                    for i, name in enumerate(label_names):
                        predictions[name].extend(output[:, i].tolist())
                        labels[name].extend(label[:, i].tolist())

            for name in label_names:
                pred, target = np.array(predictions[name]), np.array(labels[name])
                mse = np.mean((target - pred) ** 2)
                pcc = np.corrcoef(target, pred)[0][1] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                srcc = scipy.stats.spearmanr(target, pred)[0] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                
                all_mse.append(mse); all_pcc.append(pcc); all_srcc.append(srcc)
                
                if name == 'Production_Quality' and prefix == 'Valid': 
                    avg_srcc_on_valid_all.append(srcc)
                
                logging.info(f"[{dataloader.dataset.dataset_name}][{name}] MSE={mse:.4f} PCC={pcc:.4f} SRCC={srcc:.4f}")
                
                if self._use_wandb:
                    wandb.log({
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_MSE": mse,
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_PCC": pcc,
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_SRCC": srcc,
                        "epoch": self._epoch
                    })
            
            avg_mse_all = np.mean(all_mse)
            avg_pcc_all = np.mean(all_pcc)
            avg_srcc_all = np.mean(all_srcc)
            all_dataset_name = dataloader.dataset.dataset_name + '_ALL'

            logging.info(f"[{all_dataset_name}] AVG MSE={avg_mse_all:.4f} AVG PCC={avg_pcc_all:.4f} AVG SRCC={avg_srcc_all:.4f}")

            if self._use_wandb:
                wandb.log({
                    f"{prefix}/{all_dataset_name}_MSE": avg_mse_all,
                    f"{prefix}/{all_dataset_name}_PCC": avg_pcc_all,
                    f"{prefix}/{all_dataset_name}_SRCC": avg_srcc_all,
                    "epoch": self._epoch
                })
        
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
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=960)
    parser.add_argument('--covarreg_use', action='store_true')
    parser.add_argument('--covarreg_lambda', type=float, default=0.1)
    parser.add_argument('--domain_weight', type=float, default=None)
    parser.add_argument('--domain_nums', type=int, default=None)
    parser.add_argument('--domain_grouping_strategy', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    gin.external_configurable(torch.nn.modules.activation.ReLU, module='torch.nn.modules.activation')
    gin.external_configurable(torch.nn.modules.activation.SiLU, module='torch.nn.modules.activation')
    gin.parse_config_file(args.gin_path)
    
    gin.bind_parameter('TrainingLoop.covarreg_use', args.covarreg_use)
    gin.bind_parameter('TrainingLoop.covarreg_lambda', args.covarreg_lambda)
    if args.domain_weight is not None:
        gin.bind_parameter('TrainingLoop.domain_weight', args.domain_weight)
    if args.domain_nums is not None:
        gin.bind_parameter('AESFeatures.num_random_domains', args.domain_nums)
        gin.bind_parameter('DANNProjectionHead.num_domains', args.domain_nums)
        gin.bind_parameter('TrainingLoop.domain_nums', args.domain_nums)
    if args.domain_grouping_strategy is not None:
        gin.bind_parameter('AESFeatures.domain_grouping_strategy', args.domain_grouping_strategy)
        gin.bind_parameter('TrainingLoop.domain_grouping_strategy', args.domain_grouping_strategy)

    train_loop = TrainingLoop(save_path=args.save_path, use_wandb=args.use_wandb, seed=args.seed,
                              domain_grouping_strategy=args.domain_grouping_strategy,
                              domain_nums=args.domain_nums, domain_weight=args.domain_weight if args.domain_weight else 1.0)
    shutil.copyfile(args.gin_path, os.path.join(train_loop._save_path, 'config.gin'))
    train_loop.train()
    train_loop.test()
    if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    main()