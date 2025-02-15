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


def train_svm(train, test, cross_validate=None, k=0, dim_reduc=None, norms=True, balance=None, class_weights=False,
              kernel="LinearSVC",
              final_pred=False, get_coefs=False):
    """
    Function to train svm
    :param train: train data... (in panda dataframe)
    :param test: test data (itou)
    :param cross_validate: whether to perform cross validation (possible values: leave-one-out, k-fold
      and group-k-fold) if group_k-fold is chosen, each source file will be considered a group, so this is only relevant
      if sampling was performed and more than one file per class was provided
    :param k: k parameter for k-fold cross validation
    :param dim_reduc: dimensionality reduction of input data. Implemented values are pca and som.
    :param norms: perform normalisations, i.e. z-scores and L2 (default True)
    :param balance: up/downsampling strategy to use in imbalanced datasets
    :param class_weights: adjust class weights to balance imbalanced datasets, with weights inversely proportional to class
     frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    :param kernel: kernel for SVM
    :param final_pred: do the final predictions?
    :param get_coefs, if true, writes to disk (coefficients.csv) and plots the most important coefficients for each class
    :return: prints the scores, and then returns a dictionary containing the pipeline with a fitted svm model,
    and, if computed, the classification_report, confusion_matrix, list of misattributions, and final_predictions.
    """
    results = {}

    valid_cross_validate_options = {None, "leave-one-out", "k-fold", 'group-k-fold'}
    valid_dim_reduc_options = {None, 'pca'}
    valid_balance_options = {None, 'downsampling', 'upsampling', 'Tomek', 'SMOTE', 'SMOTETomek'}
    # Validate parameters
    if cross_validate not in valid_cross_validate_options:
        raise ValueError(
            f"Invalid cross-validation option: '{cross_validate}'. Valid options are {valid_cross_validate_options}.")
    if dim_reduc not in valid_dim_reduc_options:
        raise ValueError(
            f"Invalid dimensionality reduction option: '{dim_reduc}'. Valid options are {valid_dim_reduc_options}.")
    # Validate 'balance' parameter
    if balance not in valid_balance_options:
        raise ValueError(f"Invalid balance option: '{balance}'. Valid options are {valid_balance_options}.")

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
            # Adjust n_neighbors for SMOTE based on smallest class size: 
            # Ensures that the resampling method does not attempt to use more neighbors than available samples
            # in the minority class, which produced the error.
            min_class_size = min(Counter(classes).values())
            n_neighbors = min(5, min_class_size - 1)  # Default n_neighbors in SMOTE is 5
            # In case we have to temper with the n_neighbors, we print a warning message to the user
            # (might be written more clearly, but we want a short message, right?)
            if 0 < n_neighbors >= min_class_size:
                print(
                    f"Warning: Adjusting n_neighbors for SMOTE to {n_neighbors} due to small class size.")
            
            if n_neighbors == 0:
                print(
                    f"Warning: at least one class only has a single individual; cannot apply SMOTE(Tomek) due to small class size.")
            else:
            
                if balance == 'SMOTE':
                    estimators.append(('sampling', over.SMOTE(k_neighbors=n_neighbors, random_state=42)))
        
                elif balance == 'SMOTETomek':
                    estimators.append(('sampling', comb.SMOTETomek(random_state=42, smote=over.SMOTE(k_neighbors=n_neighbors, random_state=42))))

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
            if k == 0:
                k = 10 # set default

            myCV = skmodel.KFold(n_splits=k)

        if cross_validate == 'group-k-fold':
            # Get the groups as the different source texts
            works = ["_".join(t.split("_")[:-1]) for t in train.index.values]
            if k == 0:
                k = len(set(works))

            myCV = skmodel.GroupKFold(n_splits=k)

        print(".......... " + cross_validate + " cross validation will be performed ........")
        print(".......... using " + str(myCV.get_n_splits(train)) + " samples or groups........")

        # Will need to
        # 1. train a model
        # 2. get prediction
        # 3. compute score: precision, recall, F1 for all categories

        preds = skmodel.cross_val_predict(pipe, train, classes, cv=myCV, verbose=1, n_jobs=-1, groups=works)

        # and now, leave one out evaluation (very small redundancy here, one line that could be stored elsewhere)
        unique_labels = list(set(classes))

        results["confusion_matrix"] = pandas.DataFrame(metrics.confusion_matrix(classes, preds, labels=unique_labels),
                                                       index=['true:{:}'.format(x) for x in unique_labels],
                                                       columns=['pred:{:}'.format(x) for x in unique_labels])

        report = metrics.classification_report(classes, preds)
        print(report)
        results["classification_report"] = report

        # misattributions
        results["misattributions"] = pandas.DataFrame(
            [i for i in zip(list(train.index), list(classes), list(preds)) if i[1] != i[2]],
            columns=["id", "True", "Pred"]
        ).set_index('id')

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

            results["confusion_matrix"] = pandas.DataFrame(
                metrics.confusion_matrix(classes_test, preds, labels=unique_labels),
                index=['true:{:}'.format(x) for x in unique_labels],
                columns=['pred:{:}'.format(x) for x in unique_labels])

            report = metrics.classification_report(classes, preds)
            print(report)
            results["classification_report"] = report

            # misattributions
            results["misattributions"] = pandas.DataFrame(
                [i for i in zip(list(train.index), list(classes), list(preds)) if i[1] != i[2]],
                columns=["id", "True", "Pred"]
            ).set_index('id')

    # AND NOW, we need to evaluate or create the final predictions
    if final_pred:

        # Get the decision function too
        myclasses = pipe.classes_
        decs = pipe.decision_function(test)
        dists = {}
        if len(pipe.classes_) == 2:
            results["final_predictions"] = pandas.DataFrame(
                data={**{'filename': preds_index, 'author': list(preds)}, 'Decision function': decs})

        else:
            for myclass in enumerate(myclasses):
                dists[myclass[1]] = [d[myclass[0]] for d in decs]
                results["final_predictions"] = pandas.DataFrame(
                    data={**{'filename': preds_index, 'author': list(preds)}, **dists})

    if get_coefs:
        if kernel != "LinearSVC":
            print(".......... COEFS ARE ONLY IMPLEMENTED FOR linearSVC ........")

        else:
            # For “one-vs-rest” LinearSVC the attributes coef_ and intercept_ have the shape (n_classes, n_features) and
            # (n_classes,) respectively.
            # Each row of the coefficients corresponds to one of the n_classes “one-vs-rest” classifiers and similar for the
            # intercepts, in the order of the “one” class.
            if len(pipe.classes_) == 2:
                results["coefficients"] = pandas.DataFrame(pipe.named_steps['model'].coef_,
                                 index=[pipe.classes_[0]],
                                 columns=train.columns)

                plot_coefficients(pipe.named_steps['model'].coef_[0], train.columns,
                                  pipe.classes_[0] + " versus " + pipe.classes_[1])

            else:
                results["coefficients"] = pandas.DataFrame(pipe.named_steps['model'].coef_,
                                 index=pipe.classes_,
                                 columns=train.columns)

                for i in range(len(pipe.classes_)):
                    plot_coefficients(pipe.named_steps['model'].coef_[i], train.columns, pipe.classes_[i])

    results["pipeline"] = pipe

    return results

# Following function from Aneesha Bakharia
# https://aneesha.medium.com/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

def plot_coefficients(coefs, feature_names, current_class, top_features=10):
    plt.rcParams.update({'font.size': 30})  # increase font size
    top_positive_coefficients = np.argsort(coefs)[-top_features:]
    top_negative_coefficients = np.argsort(coefs)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 8))
    colors = ['red' if c < 0 else 'blue' for c in coefs[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coefs[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right',
               rotation_mode='anchor')
    plt.title("Coefficients for " + current_class)
    plt.savefig('coefs_' + current_class + '.png', bbox_inches='tight')



def plot_rolling(final_predictions, smoothing=3):
    """
    Plots the rolling stylometry results as lines of decision function values over the text.
    
    Parameters:
    final_predictions_path : str
        Path to the CSV file containing final predictions generated by the SVM pipeline.
    
    smoothing : int or None
        The window size for smoothing the curves.
        - If smoothing is None or 0, no smoothing is applied.
        - If smoothing is an integer > 0, a simple moving average with that window size is applied.
        Default is 3, which provides a slight smoothing.
    """

    # Extract the segment center from the filename
    segment_centers = []
    for fname in final_predictions['filename']:
        parts = fname.split('_')[-1].split('-')
        start = int(parts[0])
        end = int(parts[1])
        center = (start + end) / 2.0
        segment_centers.append(center)

    final_predictions['segment_center'] = segment_centers

    final_predictions['filename'] = [fname.split('_')[1] for fname in final_predictions['filename']]
    
    # Identify candidate columns
    known_cols = {'filename', 'author', 'segment_center'}
    candidate_cols = [c for c in final_predictions.columns if c not in known_cols]

    for work in final_predictions['filename'].unique():
        fpreds_work = final_predictions[final_predictions['filename'] == work]
        # Sort by segment center to ensure chronological order
        fpreds_work = fpreds_work.sort_values('segment_center')

        # Apply smoothing if requested
        if smoothing and smoothing > 0:
            for col in candidate_cols:
                fpreds_work[col] = fpreds_work[col].rolling(window=smoothing, center=True, min_periods=1).mean()

        # Plotting
        plt.figure(figsize=(24, 12))
        for col in candidate_cols:
            plt.plot(fpreds_work['segment_center'], fpreds_work[col], label=col, linewidth=2)

        plt.title('Rolling Stylometry Decision Functions Over ' + work)
        plt.xlabel('Word index (segment center)')
        plt.ylabel('Decision Function Value')
        plt.ylim(min(-2, min(fpreds_work[candidate_cols].min()) - 0.2),
                 max(1, max(fpreds_work[candidate_cols].max())) + 0.2)
        plt.legend(title='Candidate Authors', fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig('rolling_'+ work + '.png', bbox_inches='tight')

