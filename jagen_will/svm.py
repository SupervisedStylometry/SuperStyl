import sklearn.svm as sk
import sklearn.metrics as metrics
import pandas

def train_svm(train, test):
    """
    Function to train svm
    :param train: train data... (in panda dataframe)
    :param test: test data (itou)
    :return: we'll see
    """
    print(".......... loading data ........")
    # Save the classes
    classes = list(train.loc[:,'author'])
    train = train.drop(['author', 'lang'], axis=1)

    classes_test = list(test.loc[:, 'author'])
    test = test.drop(['author', 'lang'], axis=1)

    print(".......... training SVM ........")
    # let's try a standard one
    classif = sk.SVC(kernel='linear')
    classif.fit(train, classes)

    print(".......... testing SVM ........")
    preds = classif.predict(test)

    unique_labels = set(classes + classes_test)
    unique_labels = list(unique_labels)

    pandas.DataFrame(metrics.confusion_matrix(classes_test, preds, labels=unique_labels),
                           index=['true:{:}'.format(x) for x in unique_labels],
                           columns=['pred:{:}'.format(x) for x in unique_labels]).to_csv("confusion_matrix.csv")

    print(metrics.classification_report(classes_test, preds))

    return classif