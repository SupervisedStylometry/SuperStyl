import superstyl.svm
import pandas
import joblib

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', action='store', help="Path to train file", type=str)
    parser.add_argument('--test_path', action='store', help="Path to test file", type=str, required=False,
                        default=None)
    parser.add_argument('-o', action='store', help="optional prefix for output files", type=str,
                        default=None)
    parser.add_argument('--cross_validate', action='store',
                        help="perform cross validation (test_path will be used only for final prediction)."
                             "If group_k-fold is chosen, each source file will be considered a group "
                             "(only relevant if sampling was performed and more than one file per class was provided)",
                        default=None, choices=['leave-one-out', 'k-fold', 'group-k-fold'], type=str)
    parser.add_argument('--k', action='store', help="k for k-fold (default: 10 folds for k-fold; k=n groups for group-k-fold)", default=0, type=int)
    parser.add_argument('--dim_reduc', action='store', choices=['pca'], help="optional dimensionality "
                                                                             "reduction of input data", default=None)
    parser.add_argument('--norms', action='store_true', help="perform normalisations? (default: True)",
                        default=True)
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
                             "(inversely proportional to total class samples)",
                        default=False
                        )
    parser.add_argument('--kernel', action='store',
                        help="type of kernel to use (default and recommended choice is LinearSVC; "
                             "possible alternatives are linear, sigmoid, rbf and poly, as per sklearn.svm.SVC)",
                        default="LinearSVC", choices=['LinearSVC', 'linear', 'sigmoid', 'rbf', 'poly'], type=str)
    parser.add_argument('--final', action='store_true',
                        help="final analysis on unknown dataset (no evaluation)?",
                        default=False)
    parser.add_argument('--get_coefs', action='store_true',
                        help="switch to write to disk and plot the most important coefficients"
                             " for the training feats for each class",
                        default=False)
    # New arguments for rolling stylometry plotting
    parser.add_argument('--plot_rolling', action='store_true',
                        help="If final predictions are produced, also plot rolling stylometry.")
    parser.add_argument('--plot_smoothing', action='store', type=int, default=3,
                        help="Smoothing window size for rolling stylometry plot (default:3)."
                             "Set to 0 or None to disable smoothing.")

    args = parser.parse_args()

    print(".......... loading data ........")
    train = pandas.read_csv(args.train_path, index_col=0)

    if args.test_path is not None:
        test = pandas.read_csv(args.test_path, index_col=0)
    else:
        test = None

    svm = superstyl.svm.train_svm(train, test, cross_validate=args.cross_validate, k=args.k, dim_reduc=args.dim_reduc,
                                  norms=args.norms, balance=args.balance, class_weights=args.class_weights,
                                  kernel=args.kernel, final_pred=args.final, get_coefs=args.get_coefs)

    if args.o is not None:
        args.o = args.o + "_"
    else:
        args.o = ''

    if args.cross_validate is not None or (args.test_path is not None and not args.final):
        svm["confusion_matrix"].to_csv(args.o+"confusion_matrix.csv")
        svm["misattributions"].to_csv(args.o+"misattributions.csv")

    joblib.dump(svm["pipeline"], args.o+'mySVM.joblib')

    if args.final:
        print(".......... Writing final predictions to " + args.o + "FINAL_PREDICTIONS.csv ........")
        svm["final_predictions"].to_csv(args.o+"FINAL_PREDICTIONS.csv")

        # If user requested rolling stylometry plot
        if args.plot_rolling:
            print(".......... Plotting rolling stylometry ........")
            final_pred_path = args.o+"FINAL_PREDICTIONS.csv"
            smoothing = args.plot_smoothing if args.plot_smoothing is not None else 0
            superstyl.svm.plot_rolling_stylometry(final_pred_path, smoothing=smoothing)

    if args.get_coefs:
        print(".......... Writing coefficients to disk ........")
        svm["coefficients"].to_csv(args.o+"coefficients.csv")
