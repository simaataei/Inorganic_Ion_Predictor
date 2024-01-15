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
