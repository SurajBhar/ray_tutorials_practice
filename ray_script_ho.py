from functools import partial
import os
from os import path
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import ray.cloudpickle as pickle

from torchvision.datasets import ImageFolder
from sklearn.metrics import balanced_accuracy_score


# Constants
EPOCHS = 20
N_TRIALS = 20
CLASSES = 10  # StateFarm has 10 classes


def define_model():
    """
    Defines the pretrained ViT_B_16 model with a modified last linear layer and frozen base layers.
    The model-specific transforms are also obtained from the pretrained weights.

    Returns:
        nn.Module: A Vision Transformer model with a modified last layer for 10 classes.
        Callable: Data transforms specific to the pretrained model.
    """
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
    
    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Modify the final layer for 10 classes (StateFarm)
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=CLASSES)
    
    # Get the data transforms from the pretrained model
    pretrained_vit_transforms = pretrained_vit_weights.transforms()

    return pretrained_vit, pretrained_vit_transforms

def get_data_loaders(transform):
    """
    Creates the train and validation datasets for the StateFarm dataset using ImageFolder
    and the provided data transforms.

    Args:
        transform: The data transformations to apply to the dataset images.

    Returns:
        Dataloader, Dataloader: Dataloaders for training and validation sets.
    """
    train_dir = "/home/sur06423/wacv_paper/wacv_paper/data/imbalanced_v2/train"
    val_dir = "/home/sur06423/wacv_paper/wacv_paper/data/imbalanced_v2/validation"

    # Apply the model-specific transforms to the datasets
    trainset = ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)
    valset = ImageFolder(root=val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=True)

    return train_loader, val_loader


from sklearn.metrics import balanced_accuracy_score

def calculate_balanced_accuracy(y_pred, y_true, num_classes):
    """
    Calculates the balanced accuracy score using PyTorch operations.
    (y_pred == c): Creates a boolean tensor where each element is True 
    if the predicted label equals class c, and False otherwise.

    (y_true == c): Creates another boolean tensor where each element is True 
    if the true label equals class c, and False otherwise.

    &: Performs a logical AND operation between the two boolean tensors. 
    The result is a tensor where each element is True only if both conditions 
    are met: the predicted label is class c, and the true label is also class c. 
    This effectively filters out the true positives for class c.

    .sum(): Sums up the True values in the resultant tensor, which corresponds
    to the count of true positive predictions for class c.

    Args:
        y_pred (torch.Tensor): Tensor of predicted class labels( No Logits & Probabilities, only labels).
        y_true (torch.Tensor): Tensor of true class labels.
        num_classes (int): Number of classes.

    Returns:
        float: The balanced accuracy score.
    """
    correct_per_class = torch.zeros(num_classes, device=y_pred.device)
    total_per_class = torch.zeros(num_classes, device=y_pred.device)

    for c in range(num_classes):
        # The number of true positive predictions for class c. 
        # True positives are instances that are correctly identified as 
        # belonging to class c by the classifier.
        true_positives = ((y_pred == c) & (y_true == c)).sum()
        # Condition Positive: total number of instances that actually belong to class c, 
        # regardless of whether they were correctly identified by the classifier or not.
        condition_positives = (y_true == c).sum()
        
        correct_per_class[c] = true_positives.float()
        total_per_class[c] = condition_positives.float()

    # .clamp(min=1) function ensures that no value in the total_per_class tensor is less than 1
    recall_per_class = correct_per_class / total_per_class.clamp(min=1)
    balanced_accuracy = recall_per_class.mean().item()  # Convert to Python scalar for compatibility

    return balanced_accuracy

# Define the Training & Evaluation Functions
def train(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    running_loss, num_samples = 0.0, 0
    y_pred_all = []
    y_all = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * output.size(0)
        num_samples += output.size(0)
        y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
        y_pred_all.append(y_pred_class)
        y_all.append(target)

    t_average_loss = running_loss / num_samples
    # t_balanced_accuracy = calculate_balanced_accuracy(torch.cat(y_pred_all), torch.cat(y_all), CLASSES)
    t_balanced_accuracy = balanced_accuracy_score(torch.cat(y_all).cpu().numpy(), 
                                              torch.cat(y_pred_all).cpu().numpy())

    return t_balanced_accuracy, t_average_loss

def test(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    running_loss, num_samples = 0.0, 0
    y_pred_all = []
    y_all = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            running_loss += loss.item() * output.size(0)
            num_samples += output.size(0)
            y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)
            y_pred_all.append(y_pred_class)
            y_all.append(target)

    e_average_loss = running_loss / num_samples
    # e_balanced_accuracy = calculate_balanced_accuracy(torch.cat(y_pred_all), torch.cat(y_all), CLASSES)
    e_balanced_accuracy = balanced_accuracy_score(torch.cat(y_all).cpu().numpy(), 
                                              torch.cat(y_pred_all).cpu().numpy())
    return e_balanced_accuracy, e_average_loss


# Define the Trainable class for Ray Tune     
class TrainViT(tune.Trainable):
    
    def setup(self, config):
        # detect if cuda is availalbe as ray will assign GPUs if available and configured
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_vit, pretrained_vit_transforms = define_model() 
        
        self.train_loader, self.test_loader = get_data_loaders(pretrained_vit_transforms)
        self.model = pretrained_vit.to(self.device)
        
        #setup the optimiser (try Adam instead and change parameters we are tuning)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))
                
    def step(self):
        train_balanced_accuracy, train_average_loss = train(self.model, self.optimizer, self.train_loader, device=self.device)
        test_balanced_accuracy, test_average_loss = test(self.model, self.test_loader, self.device)  
        return {"train_bal_accuracy": train_balanced_accuracy, "train_loss": train_average_loss, "test_bal_accuracy": test_balanced_accuracy, "test_loss": test_average_loss}
    
    def save_checkpoint(self, checkpoint_dir):
        # checkpoint_path = path.join(checkpoint_dir, "model.pth")
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint_data, path.join(checkpoint_dir, "model.pth"))
        return checkpoint_dir   
    
    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = path.join(checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))


# Define the scheduler
asha = ASHAScheduler(
        time_attr='training_iteration',
        metric="test_balanced_accuracy",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )

import ray
ray.shutdown()
ray.init(num_cpus=24, num_gpus=0, include_dashboard=True)


config={
    "lr": tune.uniform(0.001, 0.1),
    "momentum": tune.uniform(0.1, 0.9),
}

asha = ASHAScheduler(
        time_attr='training_iteration',
        metric="test_balanced_accuracy",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )

analysis = tune.run(
    TrainViT,
    storage_path="/home/sur06423/wacv_paper/wacv_paper/ray_results",
    resources_per_trial={
        "cpu": 2,
        "gpu": 0
    },
    num_samples=10,
    checkpoint_at_end=True,
    checkpoint_freq=10,
    # keep_checkpoints_num=3,
    scheduler=asha,
    config=config)


print("Best config is:", analysis.get_best_config(metric="test_balanced_accuracy", mode='max'))
