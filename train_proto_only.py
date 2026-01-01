import argparse
import functools
import json
import logging
import os
import random
import shutil
from typing import Dict, Sequence

import dataset.dataset_AES as dataset_lib  # Use AES Dataset
import gin
import model.model_proto_only as model_lib  # Use Proto Only Model
import numpy as np
import scipy
import torch
import torch.nn as nn
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
        num_prototypes: int | None = None,
        proto_div_lambda: float = 0.0,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self._save_path = save_path
        log_path = os.path.join(save_path, 'train.log')
        logging.basicConfig(filename=log_path, level=logging.INFO, force=True)

        self._use_wandb = use_wandb
        if self._use_wandb:
            run_name = (
                f"ProtoOnly_"
                f"NP_{num_prototypes}_"
                f"PD_{proto_div_lambda}_"
                f"S_{seed}"
            )
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
                    "num_prototypes": num_prototypes,
                    "proto_div_lambda": proto_div_lambda,
                },
                dir=save_path
            )

        # Dataset setup
        dataset_cls_partial = functools.partial(dataset_cls, layer=ssl_layer)
        train_dataset = dataset_cls_partial(dataset_name='audiomos2025-track2-train_list_filter')
        
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
        if loss_type == 'mgnll':
            self._loss_fn = _multivariate_gnll_loss
        elif loss_type == 'mse':
            self._loss_fn = nn.MSELoss()
        else:
            raise ValueError(f'{loss_type=} is invalid.')

        self._covarreg_use = covarreg_use
        self._covarreg_lambda = covarreg_lambda
        self._proto_div_lambda = proto_div_lambda
        self._num_epochs = num_epochs
        self._epoch = 0
        self._best_srcc = -1
        self._all_losses = []

    def _train_once(self, batch) -> Dict[str, float]:
        # Ignore domain labels if present (AESFeatures returns 3 items)
        if len(batch) == 3:
            features, labels, _ = batch
        else:
            features, labels = batch
            
        features, labels = features.to(self._device), labels.to(self._device)

        loss_corr = torch.tensor(0.0, device=self._device)
        loss_proto_div = torch.tensor(0.0, device=self._device)

        if self._loss_type == 'mgnll':
            # No domain weight arg needed
            means, covariance = self._model(features)
            loss_main = self._loss_fn(means, labels, covariance) 

            if self._covarreg_use:
                loss_corr = _correlation_regularization_loss(covariance)
        else: 
            means = self._model(features)
            loss_main = self._loss_fn(means, labels) 
        
        # Prototype Diversity Loss
        if self._proto_div_lambda > 0 and hasattr(self._model, 'get_prototype_diversity_loss'):
            loss_proto_div = self._model.get_prototype_diversity_loss()

        total_loss = loss_main + \
                     self._covarreg_lambda * loss_corr + \
                     self._proto_div_lambda * loss_proto_div

        total_loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)
        self._optimizer.step()
        self._optimizer.zero_grad()
        
        return { 
            'total_loss': total_loss.item(), 
            'main_loss': loss_main.item(), 
            'corr_loss': loss_corr.item(),
            'proto_div_loss': loss_proto_div.item(),
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

    # def _evaluate(self, dataloaders, prefix: str):
    #     self._model.eval()
    #     avg_srcc_on_valid_all = []
    #     label_names = ['Production_Quality', 'Production_Complexity', 'Content_Enjoyment', 'Content_Usefulness']

    #     for dataloader in dataloaders:
    #         predictions = {n: [] for n in label_names}
    #         labels = {n: [] for n in label_names}
    #         all_mse, all_pcc, all_srcc = [], [], []

    #         for batch in tqdm.tqdm(dataloader, ncols=0, desc=f"{prefix} on {dataloader.dataset.dataset_name}", unit='step'):
    #             if len(batch) == 3: feature, label, _ = batch
    #             else: feature, label = batch
                
    #             feature = feature.to(self._device)
    #             with torch.no_grad():
    #                 output = self._model(feature)
    #                 if self._loss_type == 'mgnll': output, _ = output
    #                 output = output.cpu().numpy()
    #                 for i, name in enumerate(label_names):
    #                     predictions[name].extend(output[:, i].tolist())
    #                     labels[name].extend(label[:, i].tolist())

    #         for name in label_names:
    #             pred, target = np.array(predictions[name]), np.array(labels[name])
    #             mse = np.mean((target - pred) ** 2)
    #             pcc = np.corrcoef(target, pred)[0][1] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
    #             srcc = scipy.stats.spearmanr(target, pred)[0] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                
    #             all_mse.append(mse); all_pcc.append(pcc); all_srcc.append(srcc)
    #             if name == 'Production_Quality' and prefix == 'Valid': avg_srcc_on_valid_all.append(srcc)
                
    #             logging.info(f"[{dataloader.dataset.dataset_name}][{name}] MSE={mse:.4f} SRCC={srcc:.4f}")
    #             if self._use_wandb:
    #                 wandb.log({f"{prefix}/{dataloader.dataset.dataset_name}_{name}_SRCC": srcc, "epoch": self._epoch})
            
    #         if self._use_wandb:
    #             wandb.log({f"{prefix}/AVG_SRCC": np.mean(all_srcc), "epoch": self._epoch})
        
    #     if prefix == 'Valid' and avg_srcc_on_valid_all:
    #         current_srcc = np.mean(avg_srcc_on_valid_all)
    #         if current_srcc > self._best_srcc:
    #             self._best_srcc = current_srcc
    #             self.save_model('model_best_state_dict.pt', best_metric=current_srcc)
    #     self._model.train()
    def _evaluate(self, dataloaders, prefix: str):
        self._model.eval()
        avg_srcc_on_valid_all = []
        label_names = ['Production_Quality', 'Production_Complexity', 'Content_Enjoyment', 'Content_Usefulness']

        for dataloader in dataloaders:
            predictions = {n: [] for n in label_names}
            labels = {n: [] for n in label_names}
            
            # [修改] 確保列表初始化
            all_mse, all_pcc, all_srcc = [], [], []

            for batch in tqdm.tqdm(dataloader, ncols=0, desc=f"{prefix} on {dataloader.dataset.dataset_name}", unit='step'):
                if len(batch) == 3: feature, label, _ = batch
                else: feature, label = batch
                
                feature = feature.to(self._device)
                with torch.no_grad():
                    output = self._model(feature)
                    if self._loss_type == 'mgnll': output, _ = output
                    output = output.cpu().numpy()
                    for i, name in enumerate(label_names):
                        predictions[name].extend(output[:, i].tolist())
                        labels[name].extend(label[:, i].tolist())

            for name in label_names:
                pred, target = np.array(predictions[name]), np.array(labels[name])
                
                # [邏輯確認] 計算 Metrics
                mse = np.mean((target - pred) ** 2)
                pcc = np.corrcoef(target, pred)[0][1] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                srcc = scipy.stats.spearmanr(target, pred)[0] if np.std(pred) > 0 and np.std(target) > 0 else 0.0
                
                all_mse.append(mse)
                all_pcc.append(pcc)
                all_srcc.append(srcc)
                
                if name == 'Production_Quality' and prefix == 'Valid': 
                    avg_srcc_on_valid_all.append(srcc)
                
                # [修改] Log 補齊 LCC (PCC)
                logging.info(f"[{dataloader.dataset.dataset_name}][{name}] MSE={mse:.4f} LCC={pcc:.4f} SRCC={srcc:.4f}")
                
                if self._use_wandb:
                    wandb.log({
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_MSE": mse,
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_LCC": pcc,
                        f"{prefix}/{dataloader.dataset.dataset_name}_{name}_SRCC": srcc,
                        "epoch": self._epoch
                    })
            
            # [新增] 計算並 Log 平均值 (ALL)
            avg_mse_all = np.mean(all_mse)
            avg_pcc_all = np.mean(all_pcc)
            avg_srcc_all = np.mean(all_srcc)
            all_dataset_name = dataloader.dataset.dataset_name + '_ALL'

            logging.info(f"[{all_dataset_name}] AVG MSE={avg_mse_all:.4f} AVG LCC={avg_pcc_all:.4f} AVG SRCC={avg_srcc_all:.4f}")

            if self._use_wandb:
                wandb.log({
                    f"{prefix}/{all_dataset_name}_MSE": avg_mse_all,
                    f"{prefix}/{all_dataset_name}_LCC": avg_pcc_all,
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
    parser.add_argument('--gin_path', type=str, default='proto_only.gin')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=960)
    parser.add_argument('--covarreg_use', action='store_true')
    parser.add_argument('--covarreg_lambda', type=float, default=0.1)
    parser.add_argument('--num_prototypes', type=int, default=None)
    parser.add_argument('--proto_div_lambda', type=float, default=0.0)
    
    args = parser.parse_args()

    set_seed(args.seed)
    gin.external_configurable(torch.nn.modules.activation.ReLU, module='torch.nn.modules.activation')
    gin.external_configurable(torch.nn.modules.activation.SiLU, module='torch.nn.modules.activation')
    gin.parse_config_file(args.gin_path)
    
    gin.bind_parameter('TrainingLoop.covarreg_use', args.covarreg_use)
    gin.bind_parameter('TrainingLoop.covarreg_lambda', args.covarreg_lambda)
    
    if args.num_prototypes is not None:
        gin.bind_parameter('TrainingLoop.num_prototypes', args.num_prototypes)
        gin.bind_parameter('PrototypicalHead.num_prototypes', args.num_prototypes)
        
    if args.proto_div_lambda > 0:
        gin.bind_parameter('TrainingLoop.proto_div_lambda', args.proto_div_lambda)

    train_loop = TrainingLoop(
        save_path=args.save_path, 
        use_wandb=args.use_wandb, 
        seed=args.seed,
        num_prototypes=args.num_prototypes,
        proto_div_lambda=args.proto_div_lambda
    )
    shutil.copyfile(args.gin_path, os.path.join(train_loop._save_path, 'config.gin'))
    train_loop.train()
    train_loop.test()
    if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    main()