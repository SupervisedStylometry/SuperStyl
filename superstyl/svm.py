import sklearn.svm as sk
import sklearn.metrics as metrics
import sklearn.decomposition as decomp
import sklearn.preprocessing as preproc
import sklearn.pipeline as skp
import sklearn.model_selection as skmodel
import pandas
import numpy as np
import matplotlib.pyplot as plt
import imblearn.under_sampling as under
import imblearn.over_sampling as over
import imblearn.combine as comb
import imblearn.pipeline as imbp
from collections import Counter



def train_svm(train, test, cross_validate=None, k=10, dim_reduc=None, norms=True, balance=False, class_weights=False, kernel="LinearSVC",
              final_pred=False, get_coefs=False):
    """
    Function to train svm
    :param train: train data... (in panda dataframe)
    :param test: test data (itou)
    :param cross_validate: whether or not to perform cross validation (possible values: leave-one-out, k-fold
      and group-k-fold) if group_k-fold is chosen, each source file will be considered a group, so this is only relevant
      if sampling was performed and more than one file per class was provided)
    :param k: k parameter for k-fold cross validation
    :param dim_reduc: dimensionality reduction of input data. Implemented values are pca and som.
    :param norms: perform normalisations, i.e. z-scores and L2 (default True)
    :param balance: up/downsampling strategy to use in imbalanced datasets
    :param class_weights: adjust class weights to balance imbalanced datasets, with weights inversely proportional to class
     frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    :param kernel: kernel for SVM
    :param final_pred: do the final predictions?
    :param get_coefs, if true, writes to disk (coefficients.csv) and plots the most important coefficients for each class
    :return: returns a pipeline with a fitted svm model, and if possible prints evaluation and writes to disk:
    confusion_matrix.csv, misattributions.csv and (if required) FINAL_PREDICTIONS.csv
    """
    # TODO: fix n samples in SMOTE and SMOTETomek
    # ValueError: Expected n_neighbors <= n_samples,  but n_samples = 5, n_neighbors = 6
    #

    print(".......... Formatting data ........")
    # Save the classes
    classes = list(train.loc[:, 'author'])
    train = train.drop(['author', 'lang'], axis=1)

    if test is not None:
        classes_test = list(test.loc[:, 'author'])
        test = test.drop(['author', 'lang'], axis=1)
        preds_index = list(test.index)

    cw = None
    if class_weights:
        cw = "balanced"

    # CREATING PIPELINE
    print(".......... Creating pipeline according to user choices ........")
    estimators = []

    if dim_reduc == 'pca':
        print(".......... using PCA ........")
        estimators.append(('dim_reduc', decomp.PCA()))  # chosen with default
        # which is: n_components = min(n_samples, n_features)
    if norms:
        # Z-scores
        print(".......... using normalisations ........")
        estimators.append(('scaler', preproc.StandardScaler()))
        # NB: j'utilise le built-in
        # normalisation L2
        # cf. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
        estimators.append(('normalizer', preproc.Normalizer()))

    if balance is not None:
    
        print(".......... implementing strategy to solve imbalance in data ........")

        if balance == 'downsampling':
            estimators.append(('sampling', under.RandomUnderSampler(random_state=42, replacement=False)))

        # if balance == 'ENN':
        #    enn = under.EditedNearestNeighbours()
        #    train, classes = enn.fit_resample(train, classes)

        if balance == 'Tomek':
            estimators.append(('sampling', under.TomekLinks()))

        if balance == 'upsampling':
            estimators.append(('sampling', over.RandomOverSampler(random_state=42)))

        if balance in ['SMOTE', 'SMOTETomek']:
            # Adjust n_neighbors for SMOTE/SMOTETomek based on smallest class size: 
            # Ensures that the resampling method does not attempt to use more neighbors than available samples in the minority class, which produced the error.
            min_class_size = min(Counter(classes).values())
            n_neighbors = min(5, min_class_size - 1)  # Default n_neighbors in SMOTE is 5
            # In case we have to temper with the n_neighbors, we print a warning message to the user (might be written more clearly, but we want a short message, right?)
            if n_neighbors >= min_class_size:
                print(f"Warning: Adjusting n_neighbors for SMOTE / SMOTETomek to {n_neighbors} due to small class size.")
            if balance == 'SMOTE':
                estimators.append(('sampling', over.SMOTE(n_neighbors=n_neighbors, random_state=42)))
            elif balance == 'SMOTETomek':
                estimators.append(('sampling', comb.SMOTETomek(n_neighbors=n_neighbors, random_state=42)))


    print(".......... choosing SVM ........")

    if kernel == "LinearSVC":
        # try a faster one
        estimators.append(('model', sk.LinearSVC(class_weight=cw, dual="auto")))
        # classif = sk.LinearSVC()

    else:
        estimators.append(('model', sk.SVC(kernel=kernel, class_weight=cw)))
        # classif = sk.SVC(kernel=kernel)

    print(".......... Creating pipeline with steps ........")
    print(estimators)

    if 'sampling' in [k[0] for k in estimators]:
        pipe = imbp.Pipeline(estimators)

    else:
        pipe = skp.Pipeline(estimators)

    # Now, doing leave one out validation or training single SVM with train / test split

    if cross_validate is not None:
        works = None
        if cross_validate == 'leave-one-out':
            myCV = skmodel.LeaveOneOut()

        if cross_validate == 'k-fold':
            myCV = skmodel.KFold(n_splits=k)

        if cross_validate == 'group-k-fold':
            # Get the groups as the different source texts
            works = ["_".join(t.split("_")[:-1]) for t in train.index.values ]
            myCV = skmodel.GroupKFold(n_splits=len(set(works)))

        print(".......... "+ cross_validate +" cross validation will be performed ........")
        print(".......... using " + str(myCV.get_n_splits(train)) + " samples or groups........")

        # Will need to
        # 1. train a model
        # 2. get prediction
        # 3. compute score: precision, recall, F1 for all categories

        preds = skmodel.cross_val_predict(pipe, train, classes, cv=myCV, verbose=1, n_jobs=-1, groups=works)

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
        if final_pred or get_coefs:
            print(".......... Training final SVM with all train set ........")
            pipe.fit(train, classes)

        if final_pred:
            preds = pipe.predict(test)

    # And now the simple case where there is only one svm to train
    else:
        pipe.fit(train, classes)
        preds = pipe.predict(test)
        if not final_pred:
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
        if len(pipe.classes_) == 2:
            pandas.DataFrame(data={**{'filename': preds_index, 'author': list(preds)}, 'Decision function': decs}).to_csv(
                "FINAL_PREDICTIONS.csv")

        else:
            for myclass in enumerate(myclasses):
                dists[myclass[1]] = [d[myclass[0]] for d in decs]
                pandas.DataFrame(data={**{'filename': preds_index, 'author': list(preds)}, **dists}).to_csv("FINAL_PREDICTIONS.csv")

    if get_coefs:
        if kernel != "LinearSVC":
            print(".......... COEFS ARE ONLY IMPLEMENTED FOR linearSVC ........")

        else:
            # For “one-vs-rest” LinearSVC the attributes coef_ and intercept_ have the shape (n_classes, n_features) and
            # (n_classes,) respectively.
            # Each row of the coefficients corresponds to one of the n_classes “one-vs-rest” classifiers and similar for the
            # intercepts, in the order of the “one” class.
            if len(pipe.classes_) == 2:
                pandas.DataFrame(pipe.named_steps['model'].coef_,
                                 index=[pipe.classes_[0]],
                                 columns=train.columns).to_csv("coefficients.csv")

                plot_coefficients(pipe.named_steps['model'].coef_[0], train.columns, pipe.classes_[0] + " versus " + pipe.classes_[1])

            else:
                pandas.DataFrame(pipe.named_steps['model'].coef_,
                                 index=pipe.classes_,
                                 columns=train.columns).to_csv("coefficients.csv")

                # TODO: optionalise  the number of top_features… ?
                for i in range(len(pipe.classes_)):
                    plot_coefficients(pipe.named_steps['model'].coef_[i], train.columns, pipe.classes_[i])

    return pipe


# Following function from Aneesha Bakharia
# https://aneesha.medium.com/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

def plot_coefficients(coefs, feature_names, current_class, top_features=10):
    plt.rcParams.update({'font.size': 30}) #increase font size
    top_positive_coefficients = np.argsort(coefs)[-top_features:]
    top_negative_coefficients = np.argsort(coefs)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 8))
    colors = ['red' if c < 0 else 'blue' for c in coefs[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coefs[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right', rotation_mode='anchor')
    plt.title("Coefficients for "+current_class)
    plt.savefig('coefs_' + current_class + '.png', bbox_inches='tight')
    # TODO: write them to disk as CSV files
    # FOLLOW-UP: New code to write coefficients to disk as CSV 
    # I also give back a message notifying of the file creation and showing the file name.
    # First: pairing feature names with their coefficients
    coefficients_df = pandas.DataFrame({'Feature Name': feature_names, 'Coefficient': coefs})
    # Sorting the dataframe by values of coefficients in descending order
    coefficients_df = coefficients_df.reindex(coefficients_df.Coefficient.abs().sort_values(ascending=False).index)
    # Writing to CSV
    coefficients_filename = 'coefs_' + current_class + '.csv'
    coefficients_df.to_csv(coefficients_filename, index=False)
    print(f"Coefficients for {current_class} written to {coefficients_filename}")
