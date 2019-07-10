import sklearn.svm as sk
import sklearn.metrics as metrics
import sklearn.decomposition as decomp
import pandas

def train_svm(train, test, withPca=False, norms=False):
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

    if norms:
        # Z-scores
        print(".......... performing normalisations ........")
        feat_stats = pandas.DataFrame(columns=["mean", "std"])
        feat_stats.loc[:, "mean"] = list(train.mean(axis=0))
        feat_stats.loc[:, "std"] = list(train.std(axis=0))
        feat_stats.to_csv("feat_stats.csv")

        for col in list(train.columns):
            train[col] = (train[col] - train[col].mean()) / train[col].std()

        for index,col in enumerate(test.columns):
            test[:,index] = (test[:,index] - feat_stats[index,"mean"]) / feat_stats[index,"std"]


    if withPca:
        print(".......... performing PCA ........")
        pca = decomp.PCA(n_components=100)  # adjust yourself
        pca.fit(train)
        train = pca.transform(train)
        test = pca.transform(test)


    print(".......... training SVM ........")
    # let's try a standard one: only with PCA, otherwise too hard
    if withPca:
        classif = sk.SVC(kernel='linear')

    else:
        # try a faster one
        classif = sk.LinearSVC()

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