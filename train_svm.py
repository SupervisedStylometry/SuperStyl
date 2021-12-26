import superstyl.svm
import pandas
import joblib


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', action='store', help="Path to train file", type=str)
    parser.add_argument('--test_path', action='store', help="Path to test file", type=str, required=False, default=None)
    parser.add_argument('--cross_validate', action='store',
                        help="perform cross validation (test_path will be used only for final prediction)",
                        default=None, choices=['leave-one-out', 'k-fold'], type=str)
    parser.add_argument('--k', action='store', help="k for k-fold", default=10, type=int)
    parser.add_argument('--dim_reduc', action='store', choices=['pca'], help="optional dimensionality reduction of input data (warn: som is broken)", default=None)
    parser.add_argument('--norms', action='store_true', help="perform normalisations?", default=True)
    parser.add_argument('--balance', action='store', choices=["class_weight", "downsampling", "Tomek", "upsampling", "SMOTE", "SMOTETomek"],
        help="which "
            "strategy to use in case of imbalanced datasets: "
            "downsampling (random without replacement), "
            "Tomek (downs. by removing Tomek links), "
#            "ENN (EditedNearestNeighbours, downs. by removing samples close to the decision boundary), "                                                                                                                  
            "upsampling (random over sampling with replacement)"
            "SMOTE (upsampling with SMOTE - Synthetic Minority Over-sampling Technique)"
            "SMOTETomek (over+undersampling with SMOTE+Tomek)",
                        default=None)
    parser.add_argument('--class_weights', action='store_true', help="whether to use class weights in imbalanced "
                                                                     "datasets (inversely proportional to total "
                                                                     "class samples)",
                        default=False
                        )
    parser.add_argument('--kernel', action='store',
                        help="type of kernel to use (default LinearSVC; possible alternatives, linear, polynomial, rbf, sigmoid)",
                        default="LinearSVC", choices=['LinearSVC', 'linear', 'polynomial', 'rbf', 'sigmoid'], type=str)
    parser.add_argument('--final', action='store_true', help="final analysis on unknown dataset (no evaluation)?", default=False)
    parser.add_argument('--get_coefs', action='store_true', help="switch to write to disk and plots the most important coefficients for the training feats for each class",
                        default=False)
    args = parser.parse_args()

    # path = "data/feats_tests_train.csv"
    # train_path = "data/feats_tests_realForTest.csv"
    # test_path = "data/feats_tests_valid_realForTest.csv"
    print(".......... loading data ........")
    train = pandas.read_csv(args.train_path, index_col=0)

    if args.test_path is not None:
        test = pandas.read_csv(args.test_path, index_col=0)

    else:
        test = None

    svm = superstyl.svm.train_svm(train, test, cross_validate=args.cross_validate, k=args.k, dim_reduc=args.dim_reduc,
                                  norms=args.norms, balance=args.balance, class_weights=args.class_weights,
                                   kernel=args.kernel, final_pred=args.final, get_coefs=args.get_coefs)

    joblib.dump(svm, 'mySVM.joblib')
