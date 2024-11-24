import os
import ray
from ray import tune
from ray.tune import Tuner, TuneConfig, with_resources
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from ray.tune.schedulers import AsyncHyperBandScheduler

# Constants
EPOCHS = 10
CLASSES = 10  # Assume 10 classes for the StateFarm dataset

def define_model(use_gpu, num_classes=10):
    """
    Defines the pretrained ViT_B_16 model with a modified last linear layer and frozen base layers.
    """
    pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=pretrained_weights)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier for the target dataset
    # model.heads = nn.Linear(model.heads.in_features, CLASSES)
    model.heads = nn.Linear(in_features=768, out_features=num_classes)
    return model, pretrained_weights.transforms()

def get_data_loaders(transform):
    """
    Creates the train and validation dataloaders.
    """
    train_dir = "/home/sur06423/wacv_paper/wacv_paper/data/imbalanced_v2/train"
    val_dir = "/home/sur06423/wacv_paper/wacv_paper/data/imbalanced_v2/validation"
    
    trainset = ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)
    valset = ImageFolder(root=val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=True)

    return train_loader, val_loader

def calculate_balanced_accuracy(y_pred, y_true, num_classes=10):
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

# Define the Training Functions
def train_model(model, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0 
    num_samples = 0
    all_predictions = []
    all_labels = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)
        batch_predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        all_predictions.append(batch_predictions)
        all_labels.append(labels)

    train_loss = running_loss / num_samples
    train_balanced_accuracy = calculate_balanced_accuracy(torch.cat(all_predictions), torch.cat(all_labels))
    return train_loss, train_balanced_accuracy

# Define the Validation Functions
def validate_model(model, val_loader, device):
    model.eval()
    running_loss = 0.0 
    num_samples = 0
    all_predictions = []
    all_labels = []
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)
        batch_predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        all_predictions.append(batch_predictions)
        all_labels.append(labels)

    val_loss = running_loss / num_samples
    val_balanced_accuracy = calculate_balanced_accuracy(torch.cat(all_predictions), torch.cat(all_labels))
    return val_loss, val_balanced_accuracy

class NoOpScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass  # Do nothing

class TrainViT(tune.Trainable):
    def setup(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_gpu", False) else "cpu")
        self.model, transforms = define_model(config.get("use_gpu", False))
        self.model.to(self.device)
        self.train_loader, self.val_loader = get_data_loaders(transforms)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config["lr"], momentum=config["momentum"])

        # Number of epochs
        self.num_epochs = config["num_epochs"]

        # Configure optimizer
        if config["optimizer"].lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0))
        elif config["optimizer"].lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config.get("weight_decay", 0))
        elif config["optimizer"].lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0))
        else:
            raise ValueError("Unsupported optimizer: {}".format(config["optimizer"]))

        # Configure scheduler
        if config["scheduler"].lower() == 'constant':
            self.scheduler = NoOpScheduler(self.optimizer)
        elif config["scheduler"].lower() == 'cosineannealinglr':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["num_epochs"])
        elif config["scheduler"].lower() == 'lambdalr':
            lr_lambda = lambda epoch: 0.1 ** (epoch // 20)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif config["scheduler"].lower() == 'linearinterpolationlr':
            last_epoch_index = self.num_epochs - 1
            lr_lambda = lambda epoch: (1 - float(epoch) / float(last_epoch_index)) + (float(epoch) / float(last_epoch_index))* config["end_lr"] * (1/config["lr"])
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError("Unsupported scheduler: {}".format(config["scheduler"]))

    def step(self):
        train_loss, train_acc = train_model(self.model, self.optimizer, self.train_loader, self.device)
        self.scheduler.step()  # Step the learning rate scheduler
        val_loss, val_acc = validate_model(self.model, self.val_loader, self.device)
        return {"loss": train_loss, "accuracy": train_acc, "val_loss": val_loss, "val_acc": val_acc}

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


N_TRIALS = 8
# ASHA Scheduler for early stopping
scheduler = ASHAScheduler(
    metric="val_acc",
    mode="max",
    max_t=EPOCHS,
    grace_period=5,
    reduction_factor=2,
    brackets=1,
)

# Configuration for hyperparameters
config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "end_lr": tune.loguniform(1e-4, 1e-1),  # Only used if the linear interpolation scheduler is selected
    "momentum": tune.uniform(0.8, 0.99),
    "weight_decay": tune.uniform(0, 0.1),  # Optional weight decay for optimizers that support it
    "optimizer": tune.choice(["sgd", "adam", "adamw"]),
    "scheduler": tune.choice(["constant", "cosineannealinglr", "lambdalr", "linearinterpolationlr"]),
    "use_gpu": True,
    "num_epochs": EPOCHS,  # Defined constant or passed dynamically
}

# Setting up the Tuner with dynamic resource allocation
trainable_with_resources = with_resources(
    TrainViT,
    resources=lambda config: {"gpu": 1, "cpu": 2} if config.get("use_gpu", False) else {"cpu": 2}
)

tune_config = TuneConfig(
    num_samples=N_TRIALS,
    max_concurrent_trials=4  # Adjust based on the number of available GPUs
)


run_config = ray.train.RunConfig(name="ASHA_Trial_Exp_1",
                                 storage_path="/home/sur06423/wacv_paper/wacv_paper/ray_results",
                                 stop={"training_iteration": 5},
                                 checkpoint_config=ray.train.CheckpointConfig(
                                     checkpoint_frequency=2, 
                                     checkpoint_at_end=True
                                     ),                                 
)

# Initialize Ray
ray.shutdown()
ray.init(num_cpus=24, num_gpus=4, include_dashboard=True)  # Explicitly set the number of GPUs

# Define the directories to be added to LD_LIBRARY_PATH
library_paths = [
    "/usr/lib/xorg-nvidia-525.116.04/lib/x86_64-linux-gnu",
    "/usr/lib/xorg/lib/x86_64-linux-gnu",
    "/usr/lib/xorg-nvidia-535.113.01/lib/x86_64-linux-gnu"
]

# Current LD_LIBRARY_PATH from the environment
current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')

# Adding each path only if it is not already in the LD_LIBRARY_PATH
new_paths = [path for path in library_paths if path not in current_ld_library_path]

# Join all new paths with the existing LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths + [current_ld_library_path])

# Verify the update
print("Updated LD_LIBRARY_PATH:")
print(os.environ['LD_LIBRARY_PATH'])

# Create the Tuner and run the trials
tuner = Tuner(trainable_with_resources,
              param_space=config, 
              tune_config=tune_config,
              run_config=run_config 
              )
results = tuner.fit()

best_result = results.get_best_result(metric="val_acc", mode="max")
print("Best trial config: {}".format(best_result.config))
print("Best trial final validation accuracy: {}".format(best_result.metrics["val_acc"]))

ray.shutdown()