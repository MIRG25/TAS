"""Matched frozen MRI features and target-only linear probes (no meta-learning)."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from glioma_meta.training.losses import task_loss
from glioma_meta.training.runtime import class_weights, evaluate_task_isolated, make_discretizer, move_batch
from glioma_meta.training.seed_repeat import source_state_digest
from glioma_meta.training.transfer import atomic_torch_save, grouped_adamw, set_cosine_learning_rates
from glioma_meta.utils import save_json, set_seed

VERSION = 'matched_frozen_linear_probe_v1'
ARMS = ('mae', 'source-shared')
PRIMARY = {'survival':'c_index', 'who':'auc', 'idh':'auc'}


@torch.no_grad()
def extract_pair(model, loader, device):
    """Both trajectories consume the same deterministic crop, without labels."""
    model.requires_grad_(False).eval()
    initial = source_state_digest(model.state_dict())
    output = {'patient_ids':[], **{arm:[] for arm in ARMS}}
    for batch in loader:
        image = batch['image'].to(device)
        output['patient_ids'].extend(str(pid) for pid in batch['patient_id'])
        output['mae'].append(model.encoder(image, pooling='mean').float().cpu())
        output['source-shared'].append(model.encode_source_shared(image).float().cpu())
    for arm in ARMS:
        output[arm] = torch.cat(output[arm])
        if not torch.isfinite(output[arm]).all():
            raise RuntimeError('Non-finite frozen MRI features')
    if source_state_digest(model.state_dict()) != initial:
        raise RuntimeError('Frozen source parameters changed during feature extraction')
    return output


def fit_standardizer(features):
    if features.ndim != 2 or not len(features) or not torch.isfinite(features).all():
        raise ValueError('Standardization requires nonempty finite fit features')
    mean = features.float().mean(0)
    scale = features.float().std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return mean, scale


class LinearProbe(nn.Module):
    def __init__(self, feature_dim, output_dim, task, mean, scale):
        super().__init__()
        self.task = task
        self.register_buffer('mean', mean.detach().clone())
        self.register_buffer('scale', scale.detach().clone())
        self.head = nn.Linear(feature_dim, output_dim)

    def forward_task(self, features, task):
        if task != self.task:
            raise ValueError('Linear probe task mismatch')
        return self.head((features.float() - self.mean) / self.scale)


class FeatureDataset(Dataset):
    def __init__(self, cache, arm, manifest, task):
        self.features = cache[arm]
        self.ids = cache['patient_ids']
        self.task = task
        self.frame = manifest.set_index('patient_id').loc[self.ids]
        if len(self.features) != len(self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError('Feature cache has duplicate or mismatched patients')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        item = {'image':self.features[i], 'patient_id':self.ids[i]}
        if self.task == 'survival':
            item.update(survival_time=torch.tensor(float(row.survival_time)),
                        survival_event=torch.tensor(float(row.survival_event)),
                        survival_mask=torch.tensor(bool(row.get('survival_mask', True))))
        else:
            item[self.task] = torch.tensor(int(row[self.task+'_label']))
            item[self.task+'_mask'] = torch.tensor(bool(row.get(self.task+'_mask', True)))
        return item


def feature_loader(cache, arm, manifest, task, config, shuffle=False):
    generator = torch.Generator().manual_seed(int(config['training'].get('seed',2026)))
    return DataLoader(FeatureDataset(cache,arm,manifest,task),
                      batch_size=int(config['training']['batch_size']), shuffle=shuffle,
                      num_workers=0, generator=generator)


def reference(manifest, ids, task):
    if task != 'survival':
        return None
    frame = manifest.set_index('patient_id').loc[ids]
    return frame.survival_time.to_numpy(float), frame.survival_event.to_numpy(int)


def train_head(fit, validation, arm, manifest, config, task, fold, split, device, output, provenance):
    """Identical head initialization/order/hyperparameters; Validation selects epoch."""
    output = Path(output)
    set_seed(int(config['training'].get('seed',2026)))
    bins = int(config['model']['survival_bins']) if task == 'survival' else 2
    model = LinearProbe(fit[arm].shape[1], bins, task, *fit_standardizer(fit[arm])).to(device)
    initial_hash = source_state_digest(model.head.state_dict())
    train = feature_loader(fit, arm, manifest, task, config, shuffle=True)
    val = feature_loader(validation, arm, manifest, task, config)
    settings = config['adaptation']; validation_settings = config['validation']
    epochs = int(settings.get('epochs',200))
    if epochs < 1:
        raise ValueError('Probe epochs must be positive')
    optimizer = grouped_adamw([('linear_head',list(model.head.named_parameters()),
                               float(settings['head_learning_rate']))],float(settings.get('weight_decay',.05)))
    discretizer = make_discretizer(config)
    fit_frame = manifest[manifest.patient_id.isin(split['target_fit_ids'])]
    weights = class_weights(fit_frame,task,device,config) if task != 'survival' else None
    history = []; best = significant = -math.inf; stale = 0
    selection = validation_settings['selection_metrics'][task]
    ref = reference(manifest,split['target_fit_ids'],task)
    common = dict(method=VERSION,arm=arm,target_task=task,fold=fold,config=config,split=split,
                  source_tasks=provenance['source_tasks'],provenance=provenance,
                  source_parameters_unchanged=True,feature_extractor_trainable_parameters=0,
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  initial_head_sha256=initial_hash,training_seed=int(config['training'].get('seed',2026)),
                  feature_standardization='branch_specific_fit_only_population_std_constant_to_one',
                  crop_policy='single_deterministic_evaluation_crop_all_splits',
                  training_precision='fp32',outer_test_evaluated=False)
    save_json(config,output/'effective_config.json');save_json(split,output/'data_split.json')
    for epoch in range(1,epochs+1):
        set_cosine_learning_rates(optimizer,epoch,epochs,int(settings.get('warmup_epochs',0)),
                                 float(settings.get('minimum_learning_rate_factor',.01)))
        model.train(); losses=[]
        for raw in train:
            batch = move_batch(raw,device)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(task,model.forward_task(batch['image'],task),batch,discretizer,weights,
                             classification_reduction='batch-weight-mean')
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite linear probe loss')
            loss.backward()
            clip = float(settings.get('max_grad_norm',0.))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step();losses.append(float(loss.detach()))
        row = {'epoch':epoch,'train_loss':sum(losses)/len(losses)}
        if epoch % max(int(validation_settings.get('interval',1)),1) == 0 or epoch == epochs:
            metrics = evaluate_task_isolated(model,val,task,device,discretizer,save_predictions=False,
                                            survival_reference=ref,evaluation_config=config.get('evaluation',{}))
            score = metrics[selection]
            if score is None or not math.isfinite(score):
                raise RuntimeError('Non-finite Validation selection metric')
            row['validation_metrics'] = metrics
            if score > best:
                best=score
                atomic_torch_save({**common,'model_state':model.state_dict(),'epoch':epoch,
                                   'validation_metrics':metrics,'feature_dim':fit[arm].shape[1],
                                   'output_dim':bins,'survival_discretizer':discretizer.to_dict()},output/'best.pt')
            if score > significant + float(validation_settings.get('min_delta',0.)):
                significant=score;stale=0
            else:
                stale+=1
            print(f'Fold{fold} {task} {arm} epoch={epoch} loss={row["train_loss"]:.6f} '
                  f'validation_{selection}={score:.6f}',flush=True)
        history.append(row)
        save_json(history,output/'training_history.json')
        patience=int(validation_settings.get('early_stopping_patience',0))
        if patience > 0 and epoch >= int(validation_settings.get('min_epochs',1)) and stale >= patience:
            break
    saved=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    restored=load_head(saved,device)
    metrics=evaluate_task_isolated(restored,val,task,device,discretizer,
                                  output_dir=output/'validation'/task,prediction_metadata={
                                      'method':VERSION,'arm':arm,'fold':fold,'selected_epoch':saved['epoch']},
                                  survival_reference=ref,evaluation_config=config.get('evaluation',{}))
    if metrics != saved['validation_metrics']:
        raise RuntimeError('Reloaded head Validation metrics differ from selected checkpoint')
    result={k:v for k,v in common.items() if k not in ('config','split')}
    result.update(selected_epoch=saved['epoch'],selection_validation_metrics=metrics,
                  test_model_role='validation_selected_checkpoint')
    save_json(result,output/'validation_metrics.json')
    return result


def load_head(saved,device):
    if saved.get('method') != VERSION or saved.get('outer_test_evaluated') is not False:
        raise ValueError('Expected frozen linear probe Validation checkpoint')
    model=LinearProbe(saved['feature_dim'],saved['output_dim'],saved['target_task'],
                      saved['model_state']['mean'],saved['model_state']['scale'])
    model.load_state_dict(saved['model_state'],strict=True)
    return model.to(device).eval()


def test_head(cache,arm,manifest,config,task,split,device,output,analysis_sha256):
    """No fitting, standardization update or optimizer in the Test path."""
    output=Path(output)
    saved=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    model=load_head(saved,device)
    loader=feature_loader(cache,arm,manifest,task,config)
    metrics=evaluate_task_isolated(model,loader,task,device,make_discretizer(config),
             output_dir=output/'test'/task,prediction_metadata={'method':VERSION,'arm':arm,
                'fold':saved['fold'],'selected_epoch':saved['epoch'],'analysis_sha256':analysis_sha256},
             survival_reference=reference(manifest,split['target_fit_ids'],task),
             evaluation_config=config.get('evaluation',{}))
    result=dict(method=VERSION,arm=arm,target_task=task,fold=saved['fold'],selected_epoch=saved['epoch'],
                selection_validation_metrics=saved['validation_metrics'],test_metrics=metrics,
                trainable_parameters=saved['trainable_parameters'],outer_test_evaluated=True,
                test_model_role='validation_selected_checkpoint',analysis_sha256=analysis_sha256)
    save_json(result,output/'test_metrics.json')
    return result
