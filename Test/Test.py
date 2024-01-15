from Constants.constants import pth_model, file_val, file_train, file_test
import torch
from Classes.model import InorganicIonClassifier
from Classes.Data import IonDataset
from sklearn.metrics import classification_report

ion_dataset = IonDataset(file_train, file_test, file_val)

X_train, y_train = ion_dataset.X_train, ion_dataset.y_train
X_test, y_test = ion_dataset.X_test, ion_dataset.y_test



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InorganicIonClassifier(max(y_train) + 1).to(device)
model.load_state_dict(torch.load(pth_model))
targets, predictions = model.evaluate(X_test, y_test)
print(classification_report(targets, predictions))
