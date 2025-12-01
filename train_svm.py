import superstyl.svm
from superstyl.config import Config
import pandas
import joblib

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SVM models for stylometric analysis.")
    
    # Configuration file
    parser.add_argument('--config', action='store', type=str, default=None,
                        help="Path to a JSON configuration file"
    )

    # Data paths
    parser.add_argument('train_path', action='store', help="Path to train file", type=str)
    parser.add_argument('--test_path', action='store', help="Path to test file", type=str, required=False,
                        default=None)
    
    # Output options
    parser.add_argument('-o', action='store', help="optional prefix for output files", type=str,
                        default=None)

    
    # Cross-validation options
    parser.add_argument('--cross_validate', action='store',
                        help="perform cross validation (test_path will be used only for final prediction)."
                             "If group_k-fold is chosen, each source file will be considered a group "
                             "(only relevant if sampling was performed and more than one file per class was provided)",
                        default=None, choices=['leave-one-out', 'k-fold', 'group-k-fold'], type=str)
    
    parser.add_argument('--k', action='store', 
                        help="k for k-fold (default: 10 folds for k-fold; k=n groups for group-k-fold)", 
                        default=0, type=int)
    
    # Preprocessing options
    parser.add_argument('--dim_reduc', action='store', choices=['pca'], 
                        help="Dimensionality reduction of input data", default=None)
    
    parser.add_argument('--norms', action='store_true', help="perform normalisations? (default: True)",
                        default=True)
    
    # Balancing options
    parser.add_argument('--balance', action='store',
                        choices=["downsampling", "Tomek", "upsampling", "SMOTE", "SMOTETomek"],
                        help="which "
                             "strategy to use in case of imbalanced datasets: "
                             "downsampling (random without replacement), "
                             "Tomek (downs. by removing Tomek links), "
                             "upsampling (random over sampling with replacement), "
                             "SMOTE (upsampling with SMOTE), "
                             "SMOTETomek (over+undersampling with SMOTE+Tomek)",
                        default=None)
    
    parser.add_argument('--class_weights', action='store_true',
                        help="whether to use class weights in imbalanced datasets "
                             "(inversely proportional to total class samples)", default=False)
    
    parser.add_argument('--kernel', action='store',
                        help="type of kernel to use (default and recommended choice is LinearSVC; "
                             "possible alternatives are linear, sigmoid, rbf and poly, as per sklearn.svm.SVC)",
                        default="LinearSVC", choices=['LinearSVC', 'linear', 'sigmoid', 'rbf', 'poly'], type=str)
    
    # Output options
    parser.add_argument('--final', action='store_true',
                        help="final analysis on unknown dataset (no evaluation)?",
                        default=False)
    
    parser.add_argument('--get_coefs', action='store_true',
                        help="switch to write to disk and plot the most important coefficients"
                             " for the training feats for each class",
                        default=False)
    
    # Rolling stylometry plotting
    parser.add_argument('--plot_rolling', action='store_true',
                        help="If final predictions are produced, also plot rolling stylometry.")
    
    parser.add_argument('--plot_smoothing', action='store', type=int, default=3,
                        help="Smoothing window size for rolling stylometry plot (default:3)."
                             "Set to 0 or None to disable smoothing.")
    
    # Save configuration
    parser.add_argument(
        '--save_config', action='store', type=str, default=None,
        help="Save the configuration to a JSON file"
    )

    args = parser.parse_args()

    # Load data
    print(".......... loading data ........")
    train = pandas.read_csv(args.train_path, index_col=0)

    if args.test_path is not None:
        test = pandas.read_csv(args.test_path, index_col=0)
    else:
        test = None

    # Determine how to run the SVM
    if args.config:
        config = Config.from_json(args.config)
        # Override with CLI arguments if provided
        if args.cross_validate:
            config.svm.cross_validate = args.cross_validate
        if args.k:
            config.svm.k = args.k
        if args.dim_reduc:
            config.svm.dim_reduc = args.dim_reduc
        if args.balance:
            config.svm.balance = args.balance
        if args.class_weights:
            config.svm.class_weights = True
        if args.kernel != "LinearSVC":
            config.svm.kernel = args.kernel
        if args.final:
            config.svm.final_pred = True
        if args.get_coefs:
            config.svm.get_coefs = True
    else:
        config = Config.from_kwargs(
            cross_validate=args.cross_validate,
            dim_reduc=args.dim_reduc,
            norms=args.norms,
            balance=args.balance,
            class_weights=args.class_weights,
            kernel=args.kernel,
            final_pred=args.final,
            get_coefs=args.get_coefs
        )

    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"Configuration saved to {args.save_config}")

    # Train SVM
    svm = superstyl.svm.train_svm(train, test, config=config)

    # Output prefix
    prefix = f"{args.o}_" if args.o else ""

    # Save results
    if args.cross_validate or (args.test_path and not args.final):
        svm["confusion_matrix"].to_csv(f"{prefix}confusion_matrix.csv")
        svm["misattributions"].to_csv(f"{prefix}misattributions.csv")

    joblib.dump(svm["pipeline"], f"{prefix}mySVM.joblib")

    if args.final:
        print(f".......... Writing final predictions to {prefix}FINAL_PREDICTIONS.csv ........")
        svm["final_predictions"].to_csv(f"{prefix}FINAL_PREDICTIONS.csv")

        if args.plot_rolling:
            print(".......... Plotting rolling stylometry ........")
            superstyl.svm.plot_rolling(svm["final_predictions"], smoothing=args.plot_smoothing)

    if args.get_coefs:
        print(".......... Writing coefficients to disk ........")
        svm["coefficients"].to_csv(f"{prefix}coefficients.csv")