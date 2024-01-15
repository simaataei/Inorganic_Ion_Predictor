from Train.hyper_parameter_search import optimize_hyperparameters
from Constants.constants import file_train, file_test, file_val
from Classes.Data import IonDataset

def main():

    best_params, best_mcc = optimize_hyperparameters()

    print("Best Parameters:", best_params)
    print("Best MCC:", best_mcc)

if __name__ == "__main__":
    main()