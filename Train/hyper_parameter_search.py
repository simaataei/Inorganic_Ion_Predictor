import torch
import torch.optim as optim
import torch.nn as nn
from Classes.model import InorganicIonClassifier
import numpy as np
import json
import optuna
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from Classes.Data import IonDataset
from Constants.constants import file_test, file_train, file_val

# build the dataset
ion_dataset = IonDataset(file_train, file_test, file_val)

X_train, y_train = ion_dataset.X_train, ion_dataset.y_train
X_test, y_test = ion_dataset.X_test, ion_dataset.y_test
X_val, y_val = ion_dataset.X_val, ion_dataset.y_val

def objective(trial):
    best_mcc = -1
    # defining search space for the learning rate and the number of epochs
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 20, 100)

    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # intializing the model and moving it to the device
    model = InorganicIonClassifier(max(y_train) + 1).to(device)

    # defining the class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(np.array(y_train)),
                                         y=np.array(y_train))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # defining the loss function and the optimizer
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # defining warmup and weight decay
    total_steps = num_epochs * len(X_train)
    warmup_steps = 1000
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                              total_steps=total_steps, epochs=num_epochs,
                                              steps_per_epoch=len(X_train), pct_start=warmup_steps/total_steps)
    gradient_accumulation_steps = 2
    # training and validation
    for epoch in range(1, num_epochs + 1):
        model.train_epoch(optimizer, loss_function, scheduler, gradient_accumulation_steps, X_train, y_train)
        targets, predictions = model.evaluate(X_val, y_val)
        mcc = matthews_corrcoef(targets, predictions)

        # Save the best model state dictionary
        if mcc > best_mcc:
            best_mcc = mcc
            # Access the model's configuration
            config = model.config

            # Save the config in a file
            config_dict = config.to_dict()
            with open("config.json", "w") as config_file:
                json.dump(config_dict, config_file, indent=2)

            # Save the model
            torch.save(model.state_dict(), './Models/inorganic_ion_predictor.pth')

    return best_mcc

def optimize_hyperparameters():
    study = optuna.study.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_mcc = study.best_value

    return best_params, best_mcc
