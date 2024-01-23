import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer



class InorganicIonClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.config = AutoConfig.from_pretrained("Rostlab/prot_bert_bfd")
        self.num_class = num_classes
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = self.tokenizer(input, return_tensors="pt", truncation=True, max_length=1024)
        bert_rep = self.bert(input['input_ids'].to(device))  # Use to(device) instead of .cuda()
        cls_rep = bert_rep.last_hidden_state[0][0]
        class_scores = self.classifier(cls_rep)
        return F.softmax(class_scores.view(-1, self.num_class), dim=1)

    def train_epoch(self, optimizer, loss_function, scheduler, gradient_accumulation_steps, X_train, y_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train()
        all_loss = []
        for i in tqdm(range(len(X_train))):
            optimizer.zero_grad()

            # forward pass
            sample = X_train[i]
            pred = self(sample).float().to(device)  # Use .to(device) instead of .cuda()
            gold = torch.tensor([y_train[i]], dtype=torch.long).to(device)  # Use .to(device) instead of .cuda()
            loss = loss_function(pred, gold)

            # backward pass
            loss.backward()
            all_loss.append(loss.cpu().detach().numpy())

            # accumulate gradients
            if (i + 1) % gradient_accumulation_steps == 0 or i == len(X_train) - 1:
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()
        return

    def evaluate(self, X, y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        predictions = []
        targets = []
        for i in range(len(X)):
            with torch.no_grad():
                sample = X[i]
                pred = self(sample).float().to(device)
                predictions.append(pred.argmax().cpu().detach().numpy())
                targets.append(y[i])
        return targets, predictions

    def test(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            pred = self(x).float().to(device)
            prediction = pred.argmax().cpu().detach().numpy()
        return prediction
