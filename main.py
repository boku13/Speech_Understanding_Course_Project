"""
Main script that trains, validates, and evaluates
models for lie detection using AASIST framework.

Adapted from AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import numpy as np
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_RLDD_train, Dataset_RLDD_devNeval, lie_list)
from evaluation import calculate_metrics, compute_eer, calculate_tDCF_EER, calculate_lie_detection_metrics
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the lie detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define dataset related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    train_data_path = database_path / "train"
    val_data_path = database_path / "val"
    test_data_path = database_path / "test"

    # define model related paths
    model_tag = "LieDetection_{}_ep{}_bs{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Device: {}".format(device))
    if device == "cpu" and not args.allow_cpu:
        raise ValueError("GPU not detected! Use --allow_cpu to run on CPU.")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, val_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model_path = args.eval_model_weights or config.get("model_path")
        if not model_path:
            raise ValueError("Model path must be specified for evaluation")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded: {}".format(model_path))
        print("Start evaluation...")
        
        # Generate predictions
        produce_evaluation_file(eval_loader, model, device, eval_score_path)
        
        # Calculate comprehensive metrics
        metrics = calculate_lie_detection_metrics(
            eval_score_path, 
            output_file=model_tag / "detailed_evaluation_results.txt"
        )
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS:")
        print("="*50)
        print(f"  Accuracy:     {metrics['accuracy']*100:.3f}%")
        print(f"  EER:          {metrics['eer']*100:.3f}%")
        print(f"  AUC:          {metrics['auc']:.3f}")
        print(f"  F1 Score:     {metrics['f1']:.3f}")
        print(f"  Precision:    {metrics['precision']:.3f}")
        print(f"  Recall:       {metrics['recall']:.3f}")
        print("="*50 + "\n")
        
        # Write summary results to file
        with open(model_tag / "evaluation_summary.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        print("DONE.")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_val_acc = 0.
    best_val_eer = 1.0  # Lower is better for EER
    best_eval_acc = 0.
    best_eval_eer = 1.0
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("\n" + "="*80)
        print(f"EPOCH {epoch:03d}/{config['num_epochs']-1}")
        print("="*80)
        
        # Train for one epoch
        print("Training...")
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                  scheduler, config)
        
        # Evaluate on training set
        print("Evaluating on training set...")
        produce_evaluation_file(trn_loader, model, device,
                              metric_path / "train_score.txt", max_samples=500)
        
        train_metrics = calculate_lie_detection_metrics(
            metric_path / "train_score.txt",
            output_file=metric_path / f"train_metrics_epoch_{epoch}.txt"
        )
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        produce_evaluation_file(val_loader, model, device,
                              metric_path / "val_score.txt")
        
        val_metrics = calculate_lie_detection_metrics(
            metric_path / "val_score.txt",
            output_file=metric_path / f"val_metrics_epoch_{epoch}.txt"
        )
        
        val_acc = val_metrics["accuracy"]
        val_eer = val_metrics["eer"]
        
        # Print epoch summary
        print("\n" + "-"*50)
        print(f"EPOCH {epoch:03d} SUMMARY:")
        print("-"*50)
        print(f"Loss:          {running_loss:.5f}")
        print(f"TRAIN Metrics:")
        print(f"  Accuracy:    {train_metrics['accuracy']*100:.2f}%")
        print(f"  EER:         {train_metrics['eer']*100:.2f}%")
        print(f"  AUC:         {train_metrics['auc']:.3f}")
        print(f"  Class Dist:  Truthful={train_metrics.get('truthful_count', 0)}, Deceptive={train_metrics.get('deceptive_count', 0)}")
        print(f"VAL Metrics:")
        print(f"  Accuracy:    {val_metrics['accuracy']*100:.2f}%")
        print(f"  EER:         {val_metrics['eer']*100:.2f}%")
        print(f"  AUC:         {val_metrics['auc']:.3f}")
        print(f"  Class Dist:  Truthful={val_metrics.get('truthful_count', 0)}, Deceptive={val_metrics.get('deceptive_count', 0)}")
        print("-"*50)
        
        # Log metrics to TensorBoard
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("train_accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("train_eer", train_metrics["eer"] * 100, epoch)
        writer.add_scalar("train_auc", train_metrics["auc"], epoch)
        writer.add_scalar("train_f1", train_metrics["f1"], epoch)
        
        writer.add_scalar("val_accuracy", val_acc, epoch)
        writer.add_scalar("val_eer", val_eer * 100, epoch)
        writer.add_scalar("val_auc", val_metrics["auc"], epoch)
        writer.add_scalar("val_precision", val_metrics["precision"], epoch)
        writer.add_scalar("val_recall", val_metrics["recall"], epoch)
        writer.add_scalar("val_f1", val_metrics["f1"], epoch)

        # Save model if val accuracy or EER improves
        model_improved = False
        
        if best_val_acc <= val_acc:
            print(f"Best accuracy model found at epoch {epoch}")
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                     model_save_path / f"epoch_{epoch}_acc_{val_acc*100:.3f}.pth")
            model_improved = True
            
        if best_val_eer >= val_eer:
            print(f"Best EER model found at epoch {epoch}")
            best_val_eer = val_eer
            torch.save(model.state_dict(),
                     model_save_path / f"epoch_{epoch}_eer_{val_eer*100:.3f}.pth")
            model_improved = True

        # do evaluation whenever best model is renewed
        if model_improved and str_to_bool(config["eval_all_best"]):
            print("Evaluating on test set...")
            produce_evaluation_file(eval_loader, model, device, eval_score_path)
            
            # Calculate comprehensive metrics
            eval_metrics = calculate_lie_detection_metrics(
                eval_score_path,
                output_file=metric_path / f"eval_metrics_epoch_{epoch}.txt"
            )
            
            eval_acc = eval_metrics["accuracy"]
            eval_eer = eval_metrics["eer"]
            
            log_text = f"epoch{epoch:03d}, "
            improved_text = []
            
            if eval_acc > best_eval_acc:
                improved_text.append(f"best accuracy: {eval_acc*100:.2f}%")
                best_eval_acc = eval_acc
                
            if eval_eer < best_eval_eer:
                improved_text.append(f"best EER: {eval_eer*100:.2f}%")
                best_eval_eer = eval_eer
                torch.save(model.state_dict(), model_save_path / "best.pth")
            
            if improved_text:
                log_text += ", ".join(improved_text)
                print(log_text)
                f_log.write(log_text + "\n")

            print(f"Saving epoch {epoch} for SWA")
            optimizer_swa.update_swa()
            n_swa_update += 1
        
        writer.add_scalar("best_val_accuracy", best_val_acc, epoch)
        writer.add_scalar("best_val_eer", best_val_eer * 100, epoch)

    print("\n" + "="*50)
    print("TRAINING COMPLETE - FINAL EVALUATION")
    print("="*50)
    
    if n_swa_update > 0:
        print("Applying SWA...")
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    
    produce_evaluation_file(eval_loader, model, device, eval_score_path)
    
    # Calculate comprehensive metrics
    eval_metrics = calculate_lie_detection_metrics(
        eval_score_path,
        output_file=model_tag / "final_evaluation_results.txt"
    )
    
    eval_acc = eval_metrics["accuracy"]
    eval_eer = eval_metrics["eer"]
    
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write(f"Accuracy: {eval_acc*100:.3f}%, EER: {eval_eer*100:.3f}%, AUC: {eval_metrics['auc']:.3f}, F1: {eval_metrics['f1']:.3f}\n")
    f_log.close()

    torch.save(model.state_dict(), model_save_path / "swa.pth")

    if eval_acc >= best_eval_acc:
        best_eval_acc = eval_acc
    
    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(), model_save_path / "best.pth")
    
    print(f"Final Results:")
    print(f"  Best accuracy: {best_eval_acc*100:.3f}%")
    print(f"  Best EER:      {best_eval_eer*100:.3f}%")
    print("="*50)
    print("Experiment finished.")


# def get_model(model_config: Dict, device: torch.device):
#     """Define DNN model architecture"""
#     module = import_module("models.{}".format(model_config["architecture"]))
#     _model = getattr(module, "Model")
#     model = _model(model_config).to(device)
#     nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
#     print("no. model params:{}".format(nb_params))

#     return model

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    # print("no. model params:{}".format(nb_params))
    
    # Print model architecture
    # print("=== Model Architecture ===")
    # print(model)
    
    # Print model config
    # print("=== Model Configuration ===")
    # print(model_config)
    
    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / validation / evaluation"""
    # Paths to data directories
    train_data_path = database_path / "train"
    val_data_path = database_path / "val"
    test_data_path = database_path / "test"

    # Get training data
    d_label_trn, file_train = lie_list(train_data_path)
    print("no. training files:", len(file_train))
    
    # Add print statements to inspect dataset contents
    print("=== Training Dataset Inspection ===")
    print(f"First 5 file paths: {file_train[:5]}")
    print(f"Label distribution: {sum(d_label_trn.values())} labels")
    label_counts = {"Truthful": 0, "Deceptive": 0}
    for label in d_label_trn.values():
        if label == 1:
            label_counts["Truthful"] += 1
        else:
            label_counts["Deceptive"] += 1
    print(f"Label counts: {label_counts}")
    
    # Existing code continues...
    train_set = Dataset_RLDD_train(file_list=file_train, labels=d_label_trn)
    
    # Add print to inspect dataset structure
    print(f"Training dataset type: {type(train_set)}")
    print(f"Training dataset length: {len(train_set)}")
    
    # Get first sample to inspect
    if len(train_set) > 0:
        sample_x, sample_y, _= train_set[0]
        print(f"Sample input shape: {sample_x.shape}")
        print(f"Sample label: {sample_y}")

    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                          batch_size=config["batch_size"],
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True,
                          worker_init_fn=seed_worker,
                          generator=gen)

    # Get validation data
    _, file_val = lie_list(val_data_path)
    print("no. validation files:", len(file_val))

    val_set = Dataset_RLDD_devNeval(file_list=file_val)
    val_loader = DataLoader(val_set,
                          batch_size=config["batch_size"],
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

    # Get test data
    _, file_test = lie_list(test_data_path)
    print("no. test files:", len(file_test))
    
    test_set = Dataset_RLDD_devNeval(file_list=file_test)
    test_loader = DataLoader(test_set,
                           batch_size=config["batch_size"],
                           shuffle=False,
                           drop_last=False,
                           pin_memory=True)

    return trn_loader, val_loader, test_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    max_samples: int = None) -> None:
    """
    Perform evaluation and save the score to a file
    """
    model.eval()
    
    # Print evaluation info
    print("=== Evaluation Data Inspection ===")
    print(f"Dataloader length: {len(data_loader)} batches")
    print(f"Max samples: {max_samples}")
    
    fname_list = []
    score_list = []
    raw_outputs_list = []
    softmax_probs_list = []
    pred_labels_list = []
    sample_count = 0
    
    # Print detailed info for first batch only
    first_batch = True

    # Import softmax here to avoid scope issues
    from scipy.special import softmax as scipy_softmax
    
    for batch_x, utt_id, label in data_loader:
        batch_size = batch_x.size(0)
        
        # Print detailed information for first batch only
        if first_batch:
            print(f"First evaluation batch:")
            print(f"  Input shape: {batch_x.shape}")
            print(f"  Utterance IDs: {utt_id[:5]}")  # Print first 5 IDs
            print(f"  Input dtype: {batch_x.dtype}")
            print(f"  Input min: {batch_x.min().item()}, max: {batch_x.max().item()}")
            print(f"  Input mean: {batch_x.mean().item()}, std: {batch_x.std().item()}")
            first_batch = False
        
        batch_x = batch_x.to(device)

        with torch.no_grad():
            _, batch_out = model(batch_x)
            # Store raw outputs for analysis
            batch_out_np = batch_out.cpu().numpy()
            raw_outputs_list.append(batch_out_np)
            
            # Apply softmax to get probabilities
            batch_probs = scipy_softmax(batch_out_np, axis=1)
            softmax_probs_list.append(batch_probs)
            
            # Get scores for truthful class (index 1)
            batch_score = batch_probs[:, 1].ravel()
            
            # Get predicted labels using argmax of softmax probabilities
            batch_preds = np.argmax(batch_probs, axis=1)
            batch_pred_labels = ["Truthful" if pred == 1 else "Deceptive" for pred in batch_preds]
            pred_labels_list.extend(batch_pred_labels)
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        sample_count += len(utt_id)
        if max_samples and sample_count >= max_samples:
            break

    # Debug: Print score distribution to check if all predictions are the same
    scores_array = np.array(score_list)
    min_score = scores_array.min()
    max_score = scores_array.max()
    mean_score = scores_array.mean()
    std_score = scores_array.std()
    
    # Analyze raw outputs
    raw_outputs = np.vstack(raw_outputs_list)
    softmax_outputs = np.vstack(softmax_probs_list)
    
    # Calculate statistics for each output dimension
    output_means = np.mean(raw_outputs, axis=0)
    output_stds = np.std(raw_outputs, axis=0)
    output_mins = np.min(raw_outputs, axis=0)
    output_maxs = np.max(raw_outputs, axis=0)
    
    print(f"Raw output statistics:")
    print(f"  Means: {output_means}")
    print(f"  Stds:  {output_stds}")
    print(f"  Mins:  {output_mins}")
    print(f"  Maxs:  {output_maxs}")
    
    # Statistics for softmax outputs
    softmax_means = np.mean(softmax_outputs, axis=0)
    softmax_stds = np.std(softmax_outputs, axis=0)
    
    print(f"Softmax output statistics:")
    print(f"  Means: {softmax_means}")
    print(f"  Stds:  {softmax_stds}")
    
    print("fname", fname_list)
    # Save predictions with raw outputs and softmax probabilities ssdfsdf
    with open(save_path, "w") as fh:
        for i, (fn, sco, pred_label) in enumerate(zip(fname_list, score_list, pred_labels_list)):
            # Get raw output and softmax probabilities for this sample
            raw_output = raw_outputs[i]
            softmax_probs = softmax_outputs[i]

            print(f"Bruhhhhhhhhhhhh {fn} {pred_label} {sco:.6f} {raw_output[0]:.6f} {raw_output[1]:.6f} {softmax_probs[0]:.6f} {softmax_probs[1]:.6f}\n")
            
            # Format: filename pred_label truthful_score raw_output_0 raw_output_1 softmax_prob_0 softmax_prob_1
            fh.write(f"{fn} {pred_label} {sco:.6f} {raw_output[0]:.6f} {raw_output[1]:.6f} {softmax_probs[0]:.6f} {softmax_probs[1]:.6f}\n")
    
    print(f"Scores saved to {save_path}")
    print(f"Score stats - Min: {min_score:.4f}, Max: {max_score:.4f}, Mean: {mean_score:.4f}, Std: {std_score:.4f}")
    
    # Count predictions
    truthful_count = sum(1 for label in pred_labels_list if label == "Truthful")
    deceptive_count = sum(1 for label in pred_labels_list if label == "Deceptive")
    
    # Calculate percentage
    total_count = truthful_count + deceptive_count
    truthful_percent = (truthful_count / total_count * 100) if total_count > 0 else 0
    deceptive_percent = (deceptive_count / total_count * 100) if total_count > 0 else 0
    
    print(f"Predictions - Truthful: {truthful_count} ({truthful_percent:.1f}%), Deceptive: {deceptive_count} ({deceptive_percent:.1f}%)")
    
    # Check if all predictions are the same
    if truthful_count == 0 or deceptive_count == 0:
        print("WARNING: All predictions are the same class! The model is not learning to discriminate between classes.")
        print("Consider adjusting the class weights in the loss function or checking for data imbalance issues.")
    
    return


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # Count labels to check class balance
    label_counts = {0: 0, 1: 0}
    all_outputs = []
    all_labels = []

    # Print batch info for first batch
    print("=== Batch Content Inspection ===")
    print(f"Dataloader length: {len(trn_loader)} batches")
    print(f"Batch size: {config['batch_size']}")
    
    # Add debug print for first batch only
    first_batch = True
    
    # Use Mean Squared Error or Binary Cross Entropy loss for one-hot encoded labels
    criterion = torch.nn.BCELoss()# or nn.BCEWithLogitsLoss() if using logits
    
    for batch_x, key, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)  # One-hot encoded labels
        
        # Count labels in this batch (for monitoring class balance)
        for label in batch_y.cpu().numpy():
            label_index = label.argmax()  # Get the index of the max value
            label_counts[label_index] += 1
        
        # Forward pass
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        
        # Apply softmax to get probabilities
        batch_probs = torch.nn.functional.softmax(batch_out, dim=1)
        
        # Store outputs for analysis
        # Store softmax probabilities instead of raw logits
        all_outputs.append(batch_probs.detach().cpu().numpy())
        # all_outputs.append(batch_out.detach().cpu().numpy())
        all_labels.append(batch_y.detach().cpu().numpy())
        
        # Print detailed information for first batch only
        if first_batch:
            print(f"First batch:")
            print(f"  Input shape: {batch_x.shape}")
            print(f"  Label shape: {batch_y.shape}")
            print(f"  Input dtype: {batch_x.dtype}")
            print(f"  Label dtype: {batch_y.dtype}")
            print(f"  One-hot Labels: {batch_y.tolist()}")
            first_batch = False
        
        # Debug: Print model outputs for first batch
        if ii == 1:
            print(f"Model output examples (first batch):")
            for i in range(min(5, len(batch_out))):
                print(f"  Sample {i}: {batch_out[i].detach().cpu().numpy()} -> Label: {batch_y[i].tolist()}")
                
            print(f"Softmax outputs (first batch):")
            for i in range(min(5, len(batch_probs))):
                print(f"  Sample {i}: {batch_probs[i].detach().cpu().numpy()} -> Label: {batch_y[i].tolist()}")
            
        # Option 1: Calculate loss using softmax and MSE for one-hot encoded labels
        batch_loss = criterion(batch_probs, batch_y)
        
        # Option 2: Alternatively, use CrossEntropyLoss with label indices
        # label_indices = torch.argmax(batch_y, dim=1)
        # batch_loss = nn.CrossEntropyLoss()(batch_out, label_indices)
        
        # Backward pass and optimization
        running_loss += batch_loss.item() * batch_size
        
        optim.zero_grad()
        batch_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    # Print label distribution
    print(f"Training label distribution: Deceptive={label_counts[0]}, Truthful={label_counts[1]}")
    
    # Analyze model outputs
    import numpy as np
    all_outputs = np.vstack([o for o in all_outputs])
    all_labels = np.vstack([l for l in all_labels])
    
    # Calculate mean and std of outputs for each class
    deceptive_outputs = all_outputs[all_labels[:, 0] > 0.5]  # Where first column is 1 (Deceptive)
    truthful_outputs = all_outputs[all_labels[:, 1] > 0.5]   # Where second column is 1 (Truthful)
    
    if len(deceptive_outputs) > 0:
        deceptive_mean = np.mean(deceptive_outputs, axis=0)
        deceptive_std = np.std(deceptive_outputs, axis=0)
        print(f"Deceptive outputs - Mean: {deceptive_mean}, Std: {deceptive_std}")
    
    if len(truthful_outputs) > 0:
        truthful_mean = np.mean(truthful_outputs, axis=0)
        truthful_std = np.std(truthful_outputs, axis=0)
        print(f"Truthful outputs - Mean: {truthful_mean}, Std: {truthful_std}")
    
    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lie detection system")
    parser.add_argument("--config",
                      dest="config",
                      type=str,
                      help="configuration file",
                      required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                      type=int,
                      default=1234,
                      help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                      type=str,
                      default=None,
                      help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                      type=str,
                      default=None,
                      help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--allow_cpu",
                      action="store_true",
                      help="allow running on CPU if GPU is not available")
    main(parser.parse_args())