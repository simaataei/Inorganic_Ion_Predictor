import argparse
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn.functional as F
import torch.nn as nn



# create an argument parser
parser = argparse.ArgumentParser()


# add command-line arguments
parser.add_argument('input_file', type=str, help='Input FASTA file')
parser.add_argument('output_file', type=str, help='Output txt file in format of sequence ID \t prediction')


# parse the command-line arguments
args = parser.parse_args()

# check the input file format
with open(args.input_file, 'r') as f:
    first_line = f.readline()
    if not first_line.startswith('>'):
        raise ValueError('Input file is not a fasta file.')

# read the input test file
with open(args.input_file, 'r') as f:
    records = list(SeqIO.parse(f, 'fasta'))
    sequences_ids = [(str(record.seq), str(record.id)) for record in records]


# load the model
num_classes = 12
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
model = AutoModel.from_pretrained('simaataei/inorganic_ion_predictor')
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
model.to(device)



# read the label names
label_name =  {}
with open('./Dataset/Label_name_list_ICAT_uni_ident100_t10') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        label_name[d[2].strip('\n')] = d[1]
label_name = {int(k):v for k,v in label_name.items()}


# predict each protein sequence in the fasta file
results = []

for sequence, id in sequences_ids:
    print(f'Testing sequence :{id}\n')

    # preprocess the sequence
    sequence = ' '.join(sequence)
    sequence = ''.join(['X' if char in {'U', 'O', 'B', 'Z'} else char for char in sequence])
    tokenized_sequence = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=20000, truncation=True)
    input_ids = torch.tensor([tokenized_sequence['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenized_sequence['attention_mask']]).to(device)
    with torch.no_grad():
        preds = model(input_ids, attention_mask=attention_mask)
    cls_rep = preds.last_hidden_state[0][0]
    class_scores = model.classifier(cls_rep)
    predicted_class = int(F.softmax(class_scores.view(-1, num_classes), dim=1).float().to(device).argmax().cpu().detach().numpy())

    print(predicted_class)
    results.append(f'SeqID:{id},Prediction:{label_name[predicted_class]}')
    print(f'SeqID:{id} is predicted as a {label_name[predicted_class]} transporter. \n')


# save the predictions in the output file
with open(args.output_file, 'w') as f:
    for r in results:
        f.write(r + '\n')

