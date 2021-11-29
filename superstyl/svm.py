import sklearn.svm as sk
import sklearn.metrics as metrics
import sklearn.decomposition as decomp
import sklearn.preprocessing as preproc
import sklearn.pipeline as skp
import sklearn.model_selection as skmodel
import pandas
import numpy as np
import matplotlib.pyplot as plt


def train_svm(train, test, leave_one_out=False, dim_reduc=None, norms=True, kernel="LinearSVC", final_pred=False):
    """
    Function to train svm
    :param train: train data... (in panda dataframe)
    :param test: test data (itou)
    :param leave_one_out: whether or not to perform leave-one-out cross validation
    :param dim_reduc: dimensionality reduction of input data. Implemented values are pca and som.
    :param norms: perform normalisations, i.e. z-scores and L2 (default True)
    :param kernel: kernel for SVM
    :param final_pred: do the final predictions?
    :return: returns a pipeline with a fitted svm model, and if possible prints evaluation and writes to disk:
    confusion_matrix.csv, misattributions.csv and (if required) FINAL_PREDICTIONS.csv
    also write to the disk the 10 most important coefficient for each classifier, with
    a plot
    #TODO: add options to set the number of features, and control plotting
    """

    print(".......... Formatting data ........")
    # Save the classes
    classes = list(train.loc[:, 'author'])
    train = train.drop(['author', 'lang'], axis=1)

    if test is not None:
        classes_test = list(test.loc[:, 'author'])
        test = test.drop(['author', 'lang'], axis=1)
        preds_index = list(test.index)

    nfeats = train.columns.__len__()

    # CREATING PIPELINE
    print(".......... Creating pipeline according to user choices ........")
    estimators = []

    if dim_reduc == 'pca':
        print(".......... using PCA ........")
        estimators.append(('dim_reduc', decomp.PCA()))  # chosen with default
        # wich is: n_components = min(n_samples, n_features)

#    if dim_reduc == 'som':
#        print(".......... using SOM ........")  # TODO: fix SOM
#        som = minisom.MiniSom(20, 20, nfeats, sigma=0.3, learning_rate=0.5)  # initialization of 50x50 SOM
#        # TODO: set robust defaults, and calculate number of columns automatically
#        som.train_random(train.values, 100)
#        # too long to compute
#        # som.quantization_error(train)
#        print(".......... assigning SOM coordinates to texts ........")
#        train = som.quantization(train.values)
#        test = som.quantization(test.values)

    if norms:
        # Z-scores
        # TODO: me suis embeté à implémenter quelque chose qui existe
        # déjà via sklearn.preprocessing.StandardScaler()
        print(".......... using normalisations ........")
        estimators.append(('scaler', preproc.StandardScaler()))
        # scaler = preproc.StandardScaler().fit(train)
        # train = scaler.transform(train)
        # test = scaler.transform(test)
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

        estimators.append(('normalizer', preproc.Normalizer()))
        # transformer = preproc.Normalizer().fit(train)
        # train = transformer.transform(train)
        # transformer = preproc.Normalizer().fit(test)
        # test = transformer.transform(test)

    print(".......... choosing SVM ........")
    # let's try a standard one: only with PCA, otherwise too hard
    # if withPca:
    #    classif = sk.SVC(kernel='linear')

    # else:
    # try a faster one
    #    classif = sk.LinearSVC()

    if kernel == "LinearSVC":
        # try a faster one
        estimators.append(('model', sk.LinearSVC()))
        # classif = sk.LinearSVC()

    else:
        estimators.append(('model', sk.SVC(kernel=kernel)))
        # classif = sk.SVC(kernel=kernel)

    print(".......... Creating pipeline with steps ........")
    print(estimators)
    pipe = skp.Pipeline(estimators)

    # Now, doing leave one out validation or training single SVM with train / test split

    if leave_one_out:
        loo = skmodel.LeaveOneOut()
        print(".......... leave-one-out cross validation will be performed ........")
        print(".......... using " + str(loo.get_n_splits(train)) + " samples ........")

        # Will need to
        # 1. train a model
        # 2. get prediction
        # 3. compute score: precision, recall, F1 for all categories

        skmodel.cross_val_score(pipe, train, classes, cv=loo, verbose=1, n_jobs=-1)

        # Create the preds array
        preds = np.array([], dtype='<U9')
        for train_index, test_index in loo.split(train):
            # print(test_index)
            pipe.fit(train.iloc[train_index, ], [classes[i] for i in list(train_index)])
            preds = np.concatenate((preds, pipe.predict(train.iloc[test_index, ])))

        # and now, leave one out evaluation (very small redundancy here, one line that could be stored elsewhere)
        unique_labels = list(set(classes))
        pandas.DataFrame(metrics.confusion_matrix(classes, preds, labels=unique_labels),
                         index=['true:{:}'.format(x) for x in unique_labels],
                         columns=['pred:{:}'.format(x) for x in unique_labels]).to_csv("confusion_matrix.csv")

        print(metrics.classification_report(classes, preds))
        # writing misattributions
        pandas.DataFrame([i for i in zip(list(train.index), list(classes), list(preds)) if i[1] != i[2] ],
                         columns=["id", "True", "Pred"]
                         ).set_index('id').to_csv("misattributions.csv")

        # and now making the model for final preds after leave one out if necessary
        if final_pred:
            print(".......... Training final SVM with all train set ........")
            pipe.fit(train, classes)
            preds = pipe.predict(test)
            #pandas.DataFrame(data={'filename': preds_index, 'author': list(preds)}).to_csv("FINAL_PREDICTIONS.csv") # already there lower

    # And now the simple case where there is only one svm to train
    else:
        pipe.fit(train, classes)
        preds = pipe.predict(test)
        # and evaluate
        unique_labels = list(set(classes + classes_test))

        pandas.DataFrame(metrics.confusion_matrix(classes_test, preds, labels=unique_labels),
                         index=['true:{:}'.format(x) for x in unique_labels],
                         columns=['pred:{:}'.format(x) for x in unique_labels]).to_csv("confusion_matrix.csv")

        print(metrics.classification_report(classes_test, preds))

    # AND NOW, we need to evaluate or create the final predictions
    if final_pred:
        print(".......... Writing final predictions to FINAL_PREDICTIONS.csv ........")
        # Get the decision function too
        myclasses = pipe.classes_
        decs = pipe.decision_function(test)
        dists = {}
        for myclass in enumerate(myclasses):
            dists[myclass[1]] = [d[myclass[0]] for d in decs]

        pandas.DataFrame(data={**{'filename': preds_index, 'author': list(preds)}, **dists}).to_csv("FINAL_PREDICTIONS.csv")

    # For “one-vs-rest” LinearSVC the attributes coef_ and intercept_ have the shape (n_classes, n_features) and
    # (n_classes,) respectively.
    # Each row of the coefficients corresponds to one of the n_classes “one-vs-rest” classifiers and similar for the
    # intercepts, in the order of the “one” class.
    # Save coefficients for the last model
    # TODO: decide how to proceed for leave-one-out
    pandas.DataFrame(pipe.named_steps['model'].coef_,
                     index=pipe.classes_,
                     columns=train.columns).to_csv("coefficients.csv")

    # TODO: optionalise coefs
    for i in range(len(pipe.classes_)):
        plot_coefficients(pipe.named_steps['model'].coef_[i], train.columns, pipe.classes_[i])

    return pipe


# Following function from Aneesha Bakharia
# https://aneesha.medium.com/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

def plot_coefficients(coefs, feature_names, current_class, top_features=10):
    top_positive_coefficients = np.argsort(coefs)[-top_features:]
    top_negative_coefficients = np.argsort(coefs)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coefs[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coefs[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.title("Coefficients for "+current_class)
    plt.savefig('coefs_' + current_class + '.png')
    # TODO: write them to disk as CSV files
