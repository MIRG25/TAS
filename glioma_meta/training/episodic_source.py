"""Patient-disjoint first-order source adaptation with task-only PCGrad."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import numpy as np
import torch
from glioma_meta.training.adapter_basis import state_digest
from glioma_meta.training.gradient_surgery import pcgrad_project
from glioma_meta.training.source_classification import source_task_loss

OBJECTIVE = 'episodic-fomaml-pcgrad-v1'
KEY = 'source_training_objective'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def settings(config):
    src=config['source_training']; raw=src.get('episodic',{})
    if raw.get('enabled') is not True:
        raise ValueError('Episodic source training must be explicitly enabled')
    result={'enabled':True,'inner_steps':raw.get('inner_steps',3),
            'inner_learning_rate':raw.get('inner_learning_rate',.001),
            'patients_per_episode':raw.get('patients_per_episode',32)}
    for k,minimum in [('inner_steps',1),('patients_per_episode',2)]:
        if isinstance(result[k],bool) or not isinstance(result[k],int) or result[k]<minimum:
            raise ValueError(f'Invalid episodic {k}')
    lr=result['inner_learning_rate']
    if isinstance(lr,bool) or not isinstance(lr,(float,int)) or not math.isfinite(lr) or lr<=0:
        raise ValueError('Invalid episodic inner_learning_rate')
    if (src.get('refit_on_full_source_budget',True) or src.get('gradient_surgery','pcgrad')!='pcgrad'
            or src.get('mae_token_retention_weight',0) or src.get('patient_contrast',{}).get('enabled',False)
            or config['model'].get('qk_rank',0) or config['model'].get('qk_adapter',{}).get('enabled',False)
            or config['model'].get('compositional_adapter',{}).get('enabled',False)):
        raise ValueError('Episodic mode requires PCGrad, no refit/retention/contrast/QK/composition')
    return result


def episode_plan(ids,seed,fold,epoch,size=32):
    if len(ids)<2 or len(ids)!=len(set(ids)) or size<2:
        raise ValueError('Episode patients must be unique; minimum pool/block is two')
    derived=int(seed)+1000003*int(fold)+1009*int(epoch)
    shuffled=np.random.default_rng(derived).permutation(ids).tolist()
    chunks=[shuffled[i:i+size] for i in range(0,len(ids),size)]
    if len(chunks)>1 and len(chunks[-1])<2: chunks[-2].extend(chunks.pop())
    return [{'epoch':epoch,'index':i,'episode_seed':derived,'loader_seed':derived+10007*i,
             'support_ids':c[:len(c)//2],'query_ids':c[len(c)//2:]} for i,c in enumerate(chunks)]


def plan_digest(plan): return digest(plan)


def validate_labels(plan,frame,tasks):
    for row in plan:
        for role in ('support_ids','query_ids'):
            part=frame[frame.patient_id.isin(row[role])]
            if len(part)!=len(row[role]): raise ValueError('Episode contains unknown patients')
            for task in tasks:
                valid=(part.survival_time.notna() & part.survival_event.notna()) if task=='survival' else part[f'{task}_label'].notna()
                if f'{task}_mask' in part:valid &= part[f'{task}_mask'].fillna(False).astype(bool)
                if not valid.any():
                    raise ValueError(f'No effective {task} labels: epoch={row["epoch"]} episode={row["index"]} {role}')


def tensor_digest(state):
    # Training callers hold live CUDA state; hash detached host copies without
    # moving or changing the model, retaining identical bytes for CPU reuse.
    return state_digest({name:tensor.detach().cpu() for name,tensor in state.items()})


class FastLearner:
    def __init__(self,model):
        if any(p.requires_grad for p in model.encoder.parameters()):
            raise ValueError('Episodic MAE must be frozen')
        self.model=copy.deepcopy(model,{id(model.encoder):model.encoder})

    def reset(self,model,task):
        with torch.no_grad():
            for name,p in self.model.named_parameters():
                if not name.startswith('encoder.'):
                    p.copy_(dict(model.named_parameters())[name])
            for name,b in self.model.named_buffers():
                if not name.startswith('encoder.'): b.copy_(dict(model.named_buffers())[name])
        self.model.requires_grad_(False)
        self.model.shared_adapters.requires_grad_(True)
        self.model.private_adapters[task].requires_grad_(True)
        self.model.heads[task].requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self.model.train()


def _count(batch,task):
    values=batch['survival_time' if task=='survival' else task]
    return int(batch.get(f'{task}_mask',torch.ones_like(values,dtype=torch.bool)).bool().sum())


def _gradient(model,batches,task,discretizer,weights,reduction,amp,regularizer=False,reg_coefficient=1.,counter=None):
    active=[(k,p) for k,p in model.named_parameters() if p.requires_grad]
    count=sum(_count(b,task) for b in batches)
    patients=sum(len(b['image']) for b in batches)
    if not count: raise ValueError(f'No effective {task} labels in episode')
    grads={k:torch.zeros_like(p,dtype=torch.float32) for k,p in active}
    reg_grads={k:torch.zeros_like(p,dtype=torch.float32) for k,p in active}
    value=0.
    # Scaling protects FP16 activations; all accumulated gradients are unscaled FP32.
    scale=1024. if amp else 1.
    for b in batches:
        if counter is not None:counter[0]+=len(b['image'])
        n=_count(b,task)
        with torch.autocast(device_type=b['image'].device.type,dtype=torch.float16,enabled=amp):
            outputs,shared,private=model.source_components(b['image'])
            if regularizer is True:
                loss=model.shared_private_orthogonality(shared,{task:private[task]})*(len(b['image'])/patients)
            else:
                loss=source_task_loss(task,outputs[task],b,discretizer,weights.get(task),reduction)*(n/count)
            if regularizer=='both':
                reg_loss=model.shared_private_orthogonality(shared,{task:private[task]})*(len(b['image'])/patients)*reg_coefficient
        if regularizer=='both' and not torch.isfinite(reg_loss):return None,None,float('nan')
        if not torch.isfinite(loss): return (None,None,float('nan')) if regularizer=='both' else (None,float('nan'))
        values=torch.autograd.grad(loss.float()*scale,[p for _,p in active],allow_unused=True,retain_graph=regularizer=='both')
        for (name,_),g in zip(active,values):
            if g is not None:
                if not torch.isfinite(g).all(): return (None,None,float('nan')) if regularizer=='both' else (None,float('nan'))
                grads[name].add_(g.float()/scale)
        if regularizer=='both':
            rg=torch.autograd.grad(reg_loss.float()*scale,[p for _,p in active],allow_unused=True)
            for (name,_),g in zip(active,rg):
                if g is not None:
                    if not torch.isfinite(g).all():return None,None,float('nan')
                    reg_grads[name].add_(g.float()/scale)
        value+=float(loss.detach())
    if any(not torch.isfinite(g).all() for g in (*grads.values(),*reg_grads.values())):
        return (None,None,float('nan')) if regularizer=='both' else (None,float('nan'))
    return (grads,reg_grads,value) if regularizer=='both' else (grads,value)


def _clip_finite(parameters,maximum):
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):return False
    if maximum>0 and not torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters,maximum)):return False
    return all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)


def meta_step(model,fast,support,query,tasks,optimizer,config,discretizer,weights,amp=False):
    s=settings(config); src=config['source_training']; amp=bool(amp and support[0]['image'].device.type=='cuda')
    params=dict(model.named_parameters()); task_grads=[]; regs=[]; losses={}; support_losses={}
    task_weights={t:float(src.get('task_loss_weights',{}).get(t,1.)) for t in tasks}
    if any(not math.isfinite(w) or w<=0 for w in task_weights.values()): raise ValueError('Positive finite source task weights required')
    norm=sum(task_weights.values()); clip=float(src.get('max_grad_norm',0.)); inner_updates=0
    optimizer.zero_grad(set_to_none=True)
    fail={'successful':False,'inner_updates':0,'task_losses':{},'support_losses':{},'patient_forwards':0}
    counter=[0]
    def failure():
        optimizer.zero_grad(set_to_none=True)
        return {**fail,'patient_forwards':counter[0],'inner_updates':inner_updates}
    for task in tasks:
        fast.reset(model,task)
        active=[p for p in fast.model.parameters() if p.requires_grad]
        inner=torch.optim.SGD(active,lr=s['inner_learning_rate'],momentum=0,weight_decay=0)
        for _ in range(s['inner_steps']):
            inner.zero_grad(set_to_none=True)
            g,loss=_gradient(fast.model,support,task,discretizer,weights,src.get('classification_loss_reduction','fit-patient-mean'),amp,counter=counter)
            if g is None:return failure()
            for name,p in fast.model.named_parameters():
                if name in g: p.grad=g[name].to(p.dtype)
            if not _clip_finite(active,clip):return failure()
            inner.step(); inner_updates+=1
        support_losses[task]=loss
        orth=float(src.get('orthogonality_weight',.01))
        if orth:
            g,r,loss=_gradient(fast.model,query,task,discretizer,weights,src.get('classification_loss_reduction','fit-patient-mean'),amp,'both',orth,counter)
        else:
            g,loss=_gradient(fast.model,query,task,discretizer,weights,src.get('classification_loss_reduction','fit-patient-mean'),amp,counter=counter)
            r={k:torch.zeros_like(v) for k,v in (g or {}).items()}
        if g is None:return failure()
        w=task_weights[task]/norm; task_grads.append({k:v*w for k,v in g.items()});losses[task]=loss
        r={k:v*w for k,v in r.items()}
        regs.append(r)
    shared=[k for k in params if k.startswith('shared_adapters.')]
    projected,_,diagnostic=pcgrad_project([tuple(g.get(k) for k in shared) for g in task_grads])
    combined={}
    for g,r in zip(task_grads,regs):
        for k in g: combined[k]=combined.get(k,torch.zeros_like(g[k]))+g[k]+r[k]
    for k,g in zip(shared,projected): combined[k]=g+sum((r.get(k,torch.zeros_like(g)) for r in regs))
    if any(not torch.isfinite(v).all() for v in combined.values()):return failure()
    for k,g in combined.items(): params[k].grad=g.to(params[k].dtype)
    if not _clip_finite([params[k] for k in combined],clip):return failure()
    optimizer.step()
    return {'successful':True,'inner_updates':inner_updates,'task_losses':losses,
            'support_losses':support_losses,'patient_forwards':counter[0],'gradient_diagnostics':diagnostic}


def validate_checkpoint(saved,config):
    enabled=config.get('source_training',{}).get('episodic',{}).get('enabled',False)
    recorded=saved.get(KEY)
    if not enabled and recorded is None: return
    if enabled is not True or recorded!=OBJECTIVE: raise ValueError('Episodic source objective/config mismatch')
    s=settings(config); audit=saved.get('source_episodic',{})
    ids=saved['split']['fit_ids']; seed=int(config['training'].get('seed',2026)); fold=int(saved['fold'])
    epochs=int(config['source_training']['epochs'])
    plans=[episode_plan(ids,seed,fold,e,s['patients_per_episode']) for e in range(1,epochs+1)]
    encoder={k[8:]:v for k,v in saved['model_state'].items() if k.startswith('encoder.')}
    if (audit.get('settings')!=s or audit.get('fit_ids_sha256')!=digest(ids)
            or audit.get('plan_sha256')!=digest(plans) or audit.get('encoder_sha256')!=tensor_digest(encoder)
            or not audit.get('successful_updates',0) or audit.get('source_private_heads_transferred') is not False
            or tensor_digest(saved['encoder_state'])!=tensor_digest(encoder)):
        raise ValueError('Episodic source provenance/fingerprint mismatch')


def provenance(saved):
    return {KEY:saved[KEY],'source_episodic':saved['source_episodic']} if saved.get(KEY)==OBJECTIVE else {}


def validate_target_checkpoint(saved):
    source=saved.get('source_checkpoint',{})
    if not saved['config'].get('source_training',{}).get('episodic',{}).get('enabled',False) and not source.get(KEY):return
    encoder={k[8:]:v for k,v in saved['model_state'].items() if k.startswith('encoder.')}
    proxy={**source,'split':{'fit_ids':saved['split']['source_fit_ids']},'fold':saved['fold'],
           'model_state':saved['model_state'],'encoder_state':encoder}
    validate_checkpoint(proxy,saved['config'])
