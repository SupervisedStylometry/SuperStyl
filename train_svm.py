import jagen_will.svm
import pandas
import joblib


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', action='store', help="Path to train file", type=str)
    parser.add_argument('test_path', action='store', help="Path to test file", type=str)
    args = parser.parse_args()

    # path = "data/feats_tests_train.csv"
    # train_path = "data/feats_tests_realForTest.csv"
    # test_path = "data/feats_tests_valid_realForTest.csv"
    train = pandas.read_csv(args.train_path, index_col=0)
    test = pandas.read_csv(args.test_path, index_col=0)

    svm = jagen_will.svm.train_svm(train, test)

    joblib.dump(svm, 'mySVM.joblib')
