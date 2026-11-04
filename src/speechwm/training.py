"""CPU deterministic training. Evaluation-partition entries are rejected here."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .features import aggregate
from .metrics import equal_error_rate
from .models import build_model, model_input, predict_logits, state_digest


def deterministic_cpu(seed,threads=1):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)


def clone_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def evaluate_entries(model,architecture,entries,aggregations=(1,10,30)):
    loss_sum=0.;count=0;buckets={n:([],[]) for n in aggregations}
    for entry in entries:
        scores=predict_logits(model,architecture,entry["x"])
        labels=np.full(len(scores),entry["label"],dtype=np.float32)
        loss_sum+=float(nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(scores,dtype=torch.float32),torch.tensor(labels),reduction="sum"))
        count+=len(scores)
        for frames in aggregations:
            groups=aggregate(scores,entry["times"],frames)
            buckets[frames][0].extend([entry["label"]]*len(groups))
            buckets[frames][1].extend(g["score"] for g in groups)
    if not count:raise ValueError("No training/validation frames")
    return {"bce":loss_sum/count,"frames":count,
            **{f"eer_{n}":equal_error_rate(np.array(labels),np.array(scores))["eer"] for n,(labels,scores) in buckets.items()}}


def fit_model(train,validation,architecture,config,*,train_split="dev_train",validation_split="dev_validation"):
    if (train_split,validation_split) not in {("dev_train","dev_validation"),("train","validation")}:
        raise ValueError("Unsupported training/validation roles")
    if not train or not validation or any(e["split"]!=train_split for e in train) or any(e["split"]!=validation_split for e in validation):
        raise ValueError("Trainer accepts only the configured training and validation roles")
    if any(e.get("kind","main")!="main" for e in train+validation):raise ValueError("Zero controls are excluded from fitting")
    if {e["parent_utterance_id"] for e in train}&{e["parent_utterance_id"] for e in validation}:
        raise ValueError("Parent leakage into validation")
    if {e["speaker_id"] for e in train}&{e["speaker_id"] for e in validation}:
        raise ValueError("Speaker leakage into validation")
    deterministic_cpu(config["seed"],config["cpu_threads"])
    model=build_model(architecture)
    initial_hash=state_digest(model.state_dict())
    x=torch.from_numpy(np.concatenate([e["x"] for e in train]).astype(np.float32))
    y=torch.from_numpy(np.concatenate([np.full(len(e["x"]),e["label"],dtype=np.float32) for e in train]))
    optimizer=torch.optim.RMSprop(model.parameters(),**config["optimizer"])
    generator=torch.Generator(device="cpu").manual_seed(config["seed"])
    criterion=nn.BCEWithLogitsLoss();history=[];best_state=None;best_loss=float("inf");best_epoch=0
    updates=0;maximum_gradient=0.
    for epoch in range(1,config["epochs"]+1):
        model.train();order=torch.randperm(len(x),generator=generator)
        online_loss=0.;epoch_updates=0
        for start in range(0,len(x),config["batch_size"]):
            indices=order[start:start+config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(model_input(x[indices],architecture)).squeeze(1),y[indices])
            if not torch.isfinite(loss):raise ValueError("Non-finite training loss")
            loss.backward()
            gradient=sum(float(torch.sum(p.grad*p.grad)) for p in model.parameters() if p.grad is not None)**.5
            if not np.isfinite(gradient):raise ValueError("Non-finite gradient")
            maximum_gradient=max(maximum_gradient,gradient)
            optimizer.step();online_loss+=float(loss.detach())*len(indices);updates+=1;epoch_updates+=1
        train_stats=evaluate_entries(model,architecture,train)
        val_stats=evaluate_entries(model,architecture,validation)
        history.append({"epoch":epoch,"online_train_bce":online_loss/len(x),"train_bce":train_stats["bce"],
            "validation_bce":val_stats["bce"],"train_eer_1":train_stats["eer_1"],
            **{f"validation_eer_{n}":val_stats[f"eer_{n}"] for n in [1,10,30]},
            "train_frames":len(x),"positive_train_frames":int(y.sum()),"validation_frames":val_stats["frames"],
            "updates":epoch_updates,"shuffle_sha256":__import__('hashlib').sha256(order.numpy().tobytes()).hexdigest()})
        if val_stats["bce"]<best_loss:
            best_loss=val_stats["bce"];best_epoch=epoch;best_state=clone_state(model)
    final=clone_state(model)
    if initial_hash==state_digest(final) or maximum_gradient==0:raise ValueError("Model parameters did not learn")
    return {"model":model,"history":history,"final_state":final,"best_state":best_state,"best_epoch":best_epoch,
            "initial_state_sha256":initial_hash,"state_sha256":state_digest(final),"optimizer_state":optimizer.state_dict(),
            "updates":updates,"maximum_gradient_norm":maximum_gradient,"rng_state":torch.get_rng_state()}


def synthetic_sanity(seed=2026):
    results={}
    for architecture in ["A","B"]:
        deterministic_cpu(seed)
        rng=np.random.default_rng(seed);y=np.repeat([0.,1.],32).astype(np.float32)
        x=(rng.normal(0,.2,(64,63))+(2*y-1)[:,None]).astype(np.float32)
        tensor=torch.from_numpy(x);target=torch.from_numpy(y)
        model=build_model(architecture);optimizer=torch.optim.RMSprop(model.parameters(),lr=.001)
        criterion=nn.BCEWithLogitsLoss()
        initial=float(criterion(model(model_input(tensor,architecture)).squeeze(1),target).detach())
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(model_input(tensor,architecture)).squeeze(1),target)
            loss.backward();optimizer.step()
        scores=predict_logits(model,architecture,x)
        final=float(criterion(torch.tensor(scores,dtype=torch.float32),target))
        accuracy=float(np.mean((scores>=0)==y))
        if final>=initial*.2 or accuracy<.95:raise ValueError("Synthetic fit sanity failed")
        results[architecture]={"initial_bce":initial,"final_bce":final,"training_accuracy":accuracy,"steps":120,
                               "scope":"artificial separable features only; not speech performance"}
    return results
