from Train.hyper_parameter_search import optimize_hyperparameters

def main():

    best_params, best_mcc = optimize_hyperparameters()

    print("Best Parameters:", best_params)
    print("Best MCC:", best_mcc)

if __name__ == "__main__":
    main()