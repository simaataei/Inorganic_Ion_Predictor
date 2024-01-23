# Inorganic_Ion_Predictor

Inorganic ion predictor classifies the transmembrane transport proteins based on their transported specific inorganic ion(s) across the membrane. Leveraging Prot-BERT language model, each input protein sequence is predicted to transport one of 12 specific substrates: 

1. ''proton''
2. ''calcium(2+)''
3. ''potassium(1+)''
4. ''chloride''
5. ''sodium(1+)''
6. ''sulfate''
7. ''zinc(2+)''
8. ''ammonium''
9. ''nitrate''
10. ''iron(2+)''
11. ''phosphate ion''
12. ''copper(1+)''

The primary structure of protein sequences is encoded to a vector using a finetuned Prot-BERT model. This model is followed by a FNN and a softmax layer to apply classification purposes. 


## Usage:
The list of required Python packages is included in the file "requirements.txt". To install these packages, run the following command:

  ```pip install -r requirements.txt```

The program could be run using the following command:

  ```python run.py [input_fasta_file] [output_file]```

For example:

  ```python run.py Datasets/test.fasta out.txt```

  
The file "test.fasta" is the input file containing protein sequences in fasta format and "out.txt" contains the id of the test sequence followed by the prediction.

