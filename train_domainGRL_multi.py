import argparse
import functools
import json
import logging
import os
import random
import shutil
from typing import Any, Dict

import dataset.dataset_AES_multi as dataset_lib
import gin
import model.model_domainGRL_multi as model_lib
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import tqdm
import wandb
from sklearn.metrics import normalized_mutual_info_score


def set_seed(seed: int = 42) -> None:
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
    if covariance.shape[1] > 1:
        mos_covariances = covariance[:, 0, 1:] 
        penalty = torch.relu(-mos_covariances)
        return torch.mean(penalty)
    return torch.tensor(0.0, device=covariance.device)

@gin.configurable
class TrainingLoopMulti:
    """Training loop for Multi-Domain GRL."""
    def __init__(
        self,
        *,
        model: nn.Module,
        save_path: str,
        loss_type: str = 'mgnll',
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        weight_decay: float = 0.0,
        dataset_cls: torch.utils.data.dataset.Dataset = dataset_lib.AESFeaturesMulti,
        num_epochs: int = 500,
        learning_rate: float = 1e-4,
        batch_size_train: int = 64,
        ssl_layer: int = 11,
        use_wandb: bool = False,
        seed: int = 960,
        covarreg_use: bool = False,
        covarreg_lambda: float = 0.1,
        domain_weight: float = 1.0,
        domain_configs: Dict[str, Dict[str, Any]] | None = None,
        exp_tag: str = 'default',
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self._save_path = save_path
        log_path = os.path.join(save_path, 'train.log')
        logging.basicConfig(filename=log_path, level=logging.INFO, force=True)

        # 1. Dataset
        dataset_cls_partial = functools.partial(dataset_cls, layer=ssl_layer)
        train_dataset = dataset_cls_partial(dataset_name='audiomos2025-track2-train_list_filter', random_seed=seed)
        
        # 2. Config Processing
        self.final_domain_configs = domain_configs if domain_configs else {}
        if 'gt' in self.final_domain_configs:
             if hasattr(train_dataset, 'num_gt_domains'):
                 real_gt_num = train_dataset.num_gt_domains
                 print(f"[Config] Detected GT classes: {real_gt_num}")
                 self.final_domain_configs['gt']['num_classes'] = real_gt_num
        
        model_config_dict = {k: v['num_classes'] for k, v in self.final_domain_configs.items()}
        print(f"[Model Config] Domain Heads: {model_config_dict}")

        # 3. NMI Analysis
        if hasattr(train_dataset, 'get_all_domain_labels') and len(self.final_domain_configs) > 0:
            all_labels = train_dataset.get_all_domain_labels()
            keys_to_analyze = [k for k in self.final_domain_configs.keys() if k in all_labels]
            
            if len(keys_to_analyze) > 1:
                logging.info(f"[NMI] Analyzing: {keys_to_analyze}")
                nmi_matrix = np.zeros((len(keys_to_analyze), len(keys_to_analyze)))
                for i, k1 in enumerate(keys_to_analyze):
                    for j, k2 in enumerate(keys_to_analyze):
                        score = normalized_mutual_info_score(all_labels[k1], all_labels[k2])
                        nmi_matrix[i, j] = score
                logging.info(f"NMI Matrix:\n{nmi_matrix}")
                print(f"NMI Matrix:\n{nmi_matrix}")

        # 4. WandB Init
        self._use_wandb = use_wandb
        if self._use_wandb:
            run_name = f"multi_{exp_tag}_DW{domain_weight}_L{covarreg_lambda}_S{seed}"
            
            wandb.init(
                project="aes-domain-grl",
                name=run_name, 
                config={
                    "loss_type": loss_type,
                    "lr": learning_rate,
                    "seed": seed,
                    "domain_weight": domain_weight,
                    "covarreg_lambda": covarreg_lambda,
                    "domain_configs": self.final_domain_configs,
                    "exp_tag": exp_tag
                },
                dir=save_path,
                reinit=True
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

        # 5. Model Setup
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        gin.bind_parameter('DANNProjectionHeadMulti.domain_configs', model_config_dict)
        
        self._model = model(in_shape=train_dataset.features_shape).to(self._device)
        self._optimizer = optimizer(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self._loss_type = loss_type
        self._domain_weight_global = domain_weight
        self._domain_loss_fn = nn.CrossEntropyLoss()
        
        if loss_type == 'mgnll':
            self._loss_fn = _multivariate_gnll_loss
        else:
            self._loss_fn = nn.MSELoss()

        self._covarreg_use = covarreg_use
        self._covarreg_lambda = covarreg_lambda
        self._num_epochs = num_epochs
        self._epoch = 0
        self._best_srcc = -1
        self._all_losses = []

    def _train_once(self, batch) -> Dict[str, float]:
        features, labels, domain_labels_dict = batch
        features = features.to(self._device)
        labels = labels.to(self._device)
        
        if self._loss_type == 'mgnll':
            (means, covariance), domain_logits_dict = self._model(features, domain_weight=self._domain_weight_global)
            loss_main = self._loss_fn(means, labels, covariance) 
            loss_corr = _correlation_regularization_loss(covariance) if self._covarreg_use else torch.tensor(0.0).to(self._device)
        else: 
            means, domain_logits_dict = self._model(features, domain_weight=self._domain_weight_global)
            loss_main = self._loss_fn(means, labels)
            loss_corr = torch.tensor(0.0).to(self._device)

        total_domain_loss = torch.tensor(0.0, device=self._device)
        log_metrics = {}

        for name, logits in domain_logits_dict.items():
            if name in domain_labels_dict:
                targets = domain_labels_dict[name].to(self._device).long()
                d_loss = self._domain_loss_fn(logits, targets)
                w = self.final_domain_configs[name]['weight']
                total_domain_loss += (d_loss * w)
                
                acc = (logits.argmax(dim=1) == targets).float().mean()
                # 簡化 key name 以便 logging 對齊
                log_metrics[f"loss_{name}"] = d_loss.item()
                log_metrics[f"acc_{name}"] = acc.item()

        total_loss = loss_main + (self._covarreg_lambda * loss_corr) + (self._domain_weight_global * total_domain_loss)

        self._optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)
        self._optimizer.step()
        
        result = {
            'total_loss': total_loss.item(),
            'main_loss': loss_main.item(),
            'domain_loss': total_domain_loss.item(), # Renamed to align closer to single domain log style
            'corr_loss': loss_corr.item()
        }
        result.update(log_metrics)
        return result

    def train(self, valid_each_epoch: bool = True):
        self._model.train()
        while self._epoch <= self._num_epochs:
            self._all_losses = []
            for batch in tqdm.tqdm(self._train_loader, ncols=0, desc=f"Epoch {self._epoch}", unit="step"):
                self._all_losses.append(self._train_once(batch))

            if self._all_losses:
                keys = self._all_losses[0].keys()
                avg_losses = {k: np.mean([d[k] for d in self._all_losses]) for k in keys}
            else:
                avg_losses = {'total_loss': 0.0}

            # [Update] 使用與 Reference 相同的 Log 格式
            log_msg = f"Epoch {self._epoch} -"
            for name, value in avg_losses.items(): 
                log_msg += f" Avg {name}={value:.4f}"
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

            # [Update] 使用 tqdm 並對齊 Reference
            for batch in tqdm.tqdm(dataloader, ncols=0, desc=f"{prefix} on {dataloader.dataset.dataset_name}", unit='step'):
                if len(batch) == 3: feature, label, _ = batch
                else: feature, label = batch
                
                feature = feature.to(self._device)
                with torch.no_grad():
                    # Evaluate 時不需要 Domain Weight
                    output, _ = self._model(feature, domain_weight=0.0)
                    if self._loss_type == 'mgnll': output, _ = output
                    output = output.cpu().numpy()
                    for i, name in enumerate(label_names):
                        predictions[name].extend(output[:, i].tolist())
                        labels[name].extend(label[:, i].tolist())

            for name in label_names:
                pred, target = np.array(predictions[name]), np.array(labels[name])
                
                # [Update] 加入 MSE, PCC 計算與防呆
                mse = np.mean((target - pred) ** 2)
                pcc = np.corrcoef(target, pred)[0][1] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                srcc = scipy.stats.spearmanr(target, pred)[0] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                
                all_mse.append(mse); all_pcc.append(pcc); all_srcc.append(srcc)

                if name == 'Production_Quality' and prefix == 'Valid': 
                    avg_srcc_on_valid_all.append(srcc)
                
                # [Update] Logging 格式對齊
                logging.info(f"[{dataloader.dataset.dataset_name}][{name}] MSE={mse:.4f} PCC={pcc:.4f} SRCC={srcc:.4f}")
                
                if self._use_wandb:
                    wandb.log({
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_MSE": mse,
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_PCC": pcc,
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_SRCC": srcc,
                        "epoch": self._epoch
                    })

            # [Update] Log Average Metrics
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
        path = os.path.join(self._save_path, 'model_best_state_dict.pt')
        if os.path.exists(path):
            self._model.load_state_dict(torch.load(path, map_location=self._device))
            self._evaluate(self._test_loaders, 'Test')

    def save_model(self, name, best_metric):
        torch.save(self._model.state_dict(), os.path.join(self._save_path, name))
        with open(os.path.join(self._save_path, "checkpoint.json"), "w") as f:
            json.dump({"best_srcc": best_metric, "epoch": self._epoch}, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gin_path', type=str, default='configs/domainGRL_multi.gin')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=960)
    parser.add_argument('--covarreg_use', action='store_true')
    parser.add_argument('--covarreg_lambda', type=float, default=0.1)
    parser.add_argument('--domain_weight', type=float, default=1.0)
    parser.add_argument('--domain_configs_json', type=str, default=None) 
    parser.add_argument('--exp_tag', type=str, default='default')
    
    args = parser.parse_args()

    set_seed(args.seed)
    gin.external_configurable(torch.nn.modules.activation.ReLU, module='torch.nn.modules.activation')
    gin.external_configurable(torch.nn.modules.activation.SiLU, module='torch.nn.modules.activation')
    gin.parse_config_file(args.gin_path)
    
    gin.bind_parameter('TrainingLoopMulti.covarreg_use', args.covarreg_use)
    gin.bind_parameter('TrainingLoopMulti.covarreg_lambda', args.covarreg_lambda)
    gin.bind_parameter('TrainingLoopMulti.domain_weight', args.domain_weight)

    domain_configs = {}
    if args.domain_configs_json:
        try:
            domain_configs = json.loads(args.domain_configs_json)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return

    train_loop = TrainingLoopMulti(
        save_path=args.save_path, 
        use_wandb=args.use_wandb, 
        seed=args.seed,
        domain_configs=domain_configs,
        exp_tag=args.exp_tag
    )
                              
    shutil.copyfile(args.gin_path, os.path.join(train_loop._save_path, 'config.gin'))
    train_loop.train()
    train_loop.test()
    if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    main()