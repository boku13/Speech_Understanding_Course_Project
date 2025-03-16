"""
Main script that trains, validates, and evaluates
models for lie detection using AASIST framework.

Adapted from AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
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
    print("Device: {}".format(device))
    if device == "cpu" and not args.allow_cpu:
        raise ValueError("GPU not detected! Use --allow_cpu to run on CPU.")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
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
        
        print(f"Evaluation results:")
        print(f"  Accuracy: {metrics['accuracy']*100:.3f}%")
        print(f"  EER: {metrics['eer']*100:.3f}%")
        print(f"  AUC: {metrics['auc']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        
        # Write summary results to file
        with open(model_tag / "evaluation_summary.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        print("DONE.")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_acc = 0.
    best_dev_eer = 1.0  # Lower is better for EER
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
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                  scheduler, config)
        
        # Validate
        produce_evaluation_file(dev_loader, model, device,
                              metric_path / "dev_score.txt")
        
        # Calculate comprehensive metrics
        dev_metrics = calculate_lie_detection_metrics(
            metric_path / "dev_score.txt",
            output_file=metric_path / f"dev_metrics_epoch_{epoch}.txt"
        )
        
        dev_acc = dev_metrics["accuracy"]
        dev_eer = dev_metrics["eer"]
        
        print("DONE.\nLoss:{:.5f}, dev_accuracy: {:.3f}%, dev_eer: {:.3f}%, dev_auc: {:.3f}".format(
            running_loss, dev_acc * 100, dev_eer * 100, dev_metrics["auc"]))
        
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_accuracy", dev_acc, epoch)
        writer.add_scalar("dev_eer", dev_eer * 100, epoch)
        writer.add_scalar("dev_auc", dev_metrics["auc"], epoch)
        writer.add_scalar("dev_precision", dev_metrics["precision"], epoch)
        writer.add_scalar("dev_recall", dev_metrics["recall"], epoch)
        writer.add_scalar("dev_f1", dev_metrics["f1"], epoch)

        # Save model if dev accuracy or EER improves
        model_improved = False
        
        if best_dev_acc <= dev_acc:
            print("Best accuracy model found at epoch", epoch)
            best_dev_acc = dev_acc
            torch.save(model.state_dict(),
                     model_save_path / "epoch_{}_acc_{:03.3f}.pth".format(epoch, dev_acc * 100))
            model_improved = True
            
        if best_dev_eer >= dev_eer:
            print("Best EER model found at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                     model_save_path / "epoch_{}_eer_{:03.3f}.pth".format(epoch, dev_eer * 100))
            model_improved = True

        # do evaluation whenever best model is renewed
        if model_improved and str_to_bool(config["eval_all_best"]):
            produce_evaluation_file(eval_loader, model, device, eval_score_path)
            
            # Calculate comprehensive metrics
            eval_metrics = calculate_lie_detection_metrics(
                eval_score_path,
                output_file=metric_path / f"eval_metrics_epoch_{epoch}.txt"
            )
            
            eval_acc = eval_metrics["accuracy"]
            eval_eer = eval_metrics["eer"]
            
            log_text = "epoch{:03d}, ".format(epoch)
            if eval_acc > best_eval_acc:
                log_text += "best accuracy, {:.4f}%, ".format(eval_acc * 100)
                best_eval_acc = eval_acc
                
            if eval_eer < best_eval_eer:
                log_text += "best EER, {:.4f}%, ".format(eval_eer * 100)
                best_eval_eer = eval_eer
                torch.save(model.state_dict(), model_save_path / "best.pth")
                
            if len(log_text) > 0:
                print(log_text)
                f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        
        writer.add_scalar("best_dev_accuracy", best_dev_acc, epoch)
        writer.add_scalar("best_dev_eer", best_dev_eer * 100, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
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
    f_log.write("Accuracy: {:.3f}%, EER: {:.3f}%, AUC: {:.3f}, F1: {:.3f}\n".format(
        eval_acc * 100, eval_eer * 100, eval_metrics["auc"], eval_metrics["f1"]))
    f_log.close()

    torch.save(model.state_dict(), model_save_path / "swa.pth")

    if eval_acc >= best_eval_acc:
        best_eval_acc = eval_acc
    
    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(), model_save_path / "best.pth")
    
    print("Experiment finished. Best accuracy: {:.3f}%, Best EER: {:.3f}%".format(
        best_eval_acc * 100, best_eval_eer * 100))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

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

    train_set = Dataset_RLDD_train(file_list=file_train, labels=d_label_trn)
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
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            # Get scores for truthful class (index 1)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    # Save predictions
    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            # Higher score means more likely to be truthful
            pred_label = "Truthful" if sco > 0.5 else "Deceptive"
            fh.write(f"{fn} {pred_label} {sco:.6f}\n")
    
    print("Scores saved to {}".format(save_path))


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

    # set objective (Loss) functions
    # Adjust class weights based on your dataset balance
    weight = torch.FloatTensor([0.5, 0.5]).to(device)  # Equal weights for both classes
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += batch_loss.item() * batch_size
        
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

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