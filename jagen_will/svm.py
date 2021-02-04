import sklearn.svm as sk
import sklearn.metrics as metrics
import sklearn.decomposition as decomp
import sklearn.preprocessing as preproc
import pandas
import minisom

def train_svm(train, test, dim_reduc=None, norms=False, kernel="LinearSVC", final_pred=False):
    """
    Function to train svm
    :param train: train data... (in panda dataframe)
    :param test: test data (itou)
    :param dim_reduc: dimensionality reduction of input data. Implemented values are pca and som.
    :return: we'll see
    """
    print(".......... formatting data ........")
    # Save the classes
    classes = list(train.loc[:,'author'])
    train = train.drop(['author', 'lang'], axis=1)

    classes_test = list(test.loc[:, 'author'])
    test = test.drop(['author', 'lang'], axis=1)
    preds_index = list(test.index)

    nfeats = train.columns.__len__()
    
    if dim_reduc == 'pca':
        print(".......... performing PCA ........")
        pca = decomp.PCA(n_components=100)  # adjust yourself
        pca.fit(train)
        train = pca.transform(train)
        test = pca.transform(test)

    if dim_reduc == 'som':
        print(".......... training SOM ........")
        som = minisom.MiniSom(20, 20, nfeats, sigma=0.3, learning_rate=0.5)  # initialization of 50x50 SOM
        # TODO: set robust defaults, and calculate number of columns automatically
        som.train_random(train.values, 100)
        # too long to compute
        # som.quantization_error(train)
        print(".......... assigning SOM coordinates to texts ........")
        train = som.quantization(train.values)
        test = som.quantization(test.values)
    
    if norms:
        # Z-scores
        # TODO: me suis embeté à implémenter quelque chose qui existe
        # déjà via sklearn.preprocessing.StandardScaler()
        print(".......... performing normalisations ........")

        scaler = preproc.StandardScaler().fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
        # feat_stats = pandas.DataFrame(columns=["mean", "std"])
        # feat_stats.loc[:, "mean"] = list(train.mean(axis=0))
        # feat_stats.loc[:, "std"] = list(train.std(axis=0))
        # feat_stats.to_csv("feat_stats.csv")
        #
        # for col in list(train.columns):
        #     if not train[col].sum() == 0:
        #         train[col] = (train[col] - train[col].mean()) / train[col].std()
        #
        # for index, col in enumerate(test.columns):
        #     if not test.iloc[:, index].sum() == 0:
        #         # keep same as train if possible
        #         if not feat_stats.loc[index,"mean"] == 0 and not feat_stats.loc[index,"std"] == 0:
        #             test.iloc[:,index] = (test.iloc[:,index] - feat_stats.loc[index,"mean"]) / feat_stats.loc[index,"std"]
        #
        #         else:
        #             test.iloc[:, index] = (test.iloc[:, index] - test.iloc[:, index].mean()) / test.iloc[:, index].std()

        # NB: je ne refais pas la meme erreur, et cette fois j'utilise le built-in
        # normalisation L2
        # cf. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer

        transformer = preproc.Normalizer().fit(train)
        train = transformer.transform(train)
        transformer = preproc.Normalizer().fit(test)
        test = transformer.transform(test)

    print(".......... training SVM ........")
    # let's try a standard one: only with PCA, otherwise too hard
    #if withPca:
    #    classif = sk.SVC(kernel='linear')

    #else:
        # try a faster one
    #    classif = sk.LinearSVC()

    if kernel == "LinearSVC":
    # try a faster one
        classif = sk.LinearSVC()

    else:
        classif = sk.SVC(kernel=kernel)


    classif.fit(train, classes)

    print(".......... testing SVM ........")
    preds = classif.predict(test)

    if final_pred:
        pandas.DataFrame(data={'filename': preds_index, 'author': list(preds)}).to_csv("FINAL_PREDICTIONS.csv")

    else:
        unique_labels = set(classes + classes_test)
        unique_labels = list(unique_labels)

        pandas.DataFrame(metrics.confusion_matrix(classes_test, preds, labels=unique_labels),
                               index=['true:{:}'.format(x) for x in unique_labels],
                               columns=['pred:{:}'.format(x) for x in unique_labels]).to_csv("confusion_matrix.csv")

        print(metrics.classification_report(classes_test, preds))
        #print(metrics.accuracy_score(classes_test, preds))

    return classif
