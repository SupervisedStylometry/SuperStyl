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
from typing import Optional, Dict, Any

from superstyl.config import Config


def train_svm(
    train: pandas.DataFrame,
    test: Optional[pandas.DataFrame] = None,
    config: Optional[Config] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train SVM model for stylometric analysis.
    
    Can be called with:
    1. A Config object: train_svm(train, test, config=my_config)
    2. Individual parameters (backward compatible):
       train_svm(train, test, cross_validate="k-fold", k=10)
    
    Args:
        train: Training data (pandas DataFrame)
        test: Test data (optional)
        config: Configuration object. If None, built from kwargs.
        **kwargs: Individual parameters for backward compatibility.
                  Supported: cross_validate, k, dim_reduc, norms, balance,
                  class_weights, kernel, final_pred, get_coefs
    
    Returns:
        Dictionary containing: pipeline, and optionally confusion_matrix,
        classification_report, misattributions, final_predictions, coefficients
    """
    # Build config from kwargs if not provided
    if config is None:
        config = Config.from_kwargs(**kwargs)
    
    # Extract SVM parameters from config
    cross_validate = config.svm.cross_validate
    k = config.svm.k
    dim_reduc = config.svm.dim_reduc
    norms = config.svm.norms
    balance = config.svm.balance
    class_weights = config.svm.class_weights
    kernel = config.svm.kernel
    final_pred = config.svm.final_pred
    get_coefs = config.svm.get_coefs

    results = {}

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
        estimators.append(('dim_reduc', decomp.PCA()))

    if norms:
        print(".......... using normalisations ........")
        estimators.append(('scaler', preproc.StandardScaler()))
        estimators.append(('normalizer', preproc.Normalizer()))

    if balance is not None:
        print(".......... implementing strategy to solve imbalance in data ........")

        if balance == 'downsampling':
            estimators.append(('sampling', under.RandomUnderSampler(random_state=42, replacement=False)))

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
                print(f"Warning: Adjusting n_neighbors for SMOTE to {n_neighbors} due to small class size.")
            
            if n_neighbors == 0:
                print("Warning: at least one class only has a single individual; cannot apply SMOTE(Tomek).")
            else:
                if balance == 'SMOTE':
                    estimators.append(('sampling', over.SMOTE(k_neighbors=n_neighbors, random_state=42)))
                elif balance == 'SMOTETomek':
                    estimators.append(('sampling', comb.SMOTETomek(
                        random_state=42,
                        smote=over.SMOTE(k_neighbors=n_neighbors, random_state=42)
                    )))

    print(".......... choosing SVM ........")

    if kernel == "LinearSVC":
        estimators.append(('model', sk.LinearSVC(class_weight=cw, dual="auto")))
    else:
        estimators.append(('model', sk.SVC(kernel=kernel, class_weight=cw)))

    print(".......... Creating pipeline with steps ........")
    print(estimators)

    if 'sampling' in [k[0] for k in estimators]:
        pipe = imbp.Pipeline(estimators)
    else:
        pipe = skp.Pipeline(estimators)

    # Cross validation or train/test split
    if cross_validate is not None:
        works = None
        if cross_validate == 'leave-one-out':
            myCV = skmodel.LeaveOneOut()

        if cross_validate == 'k-fold':
            if k == 0:
                k = 10
            myCV = skmodel.KFold(n_splits=k)

        if cross_validate == 'group-k-fold':
            works = ["_".join(t.split("_")[:-1]) for t in train.index.values]
            if k == 0:
                k = len(set(works))
            myCV = skmodel.GroupKFold(n_splits=k)

        print(f".......... {cross_validate} cross validation will be performed ........")
        print(f".......... using {myCV.get_n_splits(train)} samples or groups........")

        preds = skmodel.cross_val_predict(pipe, train, classes, cv=myCV, verbose=1, n_jobs=-1, groups=works)

        # and now, leave one out evaluation (very small redundancy here, one line that could be stored elsewhere)
        unique_labels = list(set(classes))

        results["confusion_matrix"] = pandas.DataFrame(
            metrics.confusion_matrix(classes, preds, labels=unique_labels),
            index=[f'true:{x}' for x in unique_labels],
            columns=[f'pred:{x}' for x in unique_labels]
        )

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
                index=[f'true:{x}' for x in unique_labels],
                columns=[f'pred:{x}' for x in unique_labels]
            )

            report = metrics.classification_report(classes, preds)
            print(report)
            results["classification_report"] = report

            # misattributions
            results["misattributions"] = pandas.DataFrame(
                [i for i in zip(list(train.index), list(classes), list(preds)) if i[1] != i[2]],
                columns=["id", "True", "Pred"]
            ).set_index('id')

    # Final predictions
    if final_pred:

        # Get the decision function too
        myclasses = pipe.classes_
        decs = pipe.decision_function(test)
        dists = {}
        
        if len(pipe.classes_) == 2:
            results["final_predictions"] = pandas.DataFrame(
                data={**{'filename': preds_index, 'author': list(preds)}, 'Decision function': decs}
            )
        else:
            for myclass in enumerate(myclasses):
                dists[myclass[1]] = [d[myclass[0]] for d in decs]
            results["final_predictions"] = pandas.DataFrame(
                data={**{'filename': preds_index, 'author': list(preds)}, **dists}
            )

    if get_coefs:
        if kernel != "LinearSVC":
            print(".......... COEFS ARE ONLY IMPLEMENTED FOR linearSVC ........")
        else:
            # For “one-vs-rest” LinearSVC the attributes coef_ and intercept_ have the shape (n_classes, n_features) and
            # (n_classes,) respectively.
            # Each row of the coefficients corresponds to one of the n_classes “one-vs-rest” classifiers and similar for the
            # intercepts, in the order of the “one” class.
            if len(pipe.classes_) == 2:
                results["coefficients"] = pandas.DataFrame(
                    pipe.named_steps['model'].coef_,
                    index=[pipe.classes_[0]],
                    columns=train.columns
                )
                plot_coefficients(
                    pipe.named_steps['model'].coef_[0],
                    train.columns,
                    f"{pipe.classes_[0]} versus {pipe.classes_[1]}"
                )
            else:
                results["coefficients"] = pandas.DataFrame(
                    pipe.named_steps['model'].coef_,
                    index=pipe.classes_,
                    columns=train.columns
                )
                for i in range(len(pipe.classes_)):
                    plot_coefficients(pipe.named_steps['model'].coef_[i], train.columns, pipe.classes_[i])

    results["pipeline"] = pipe

    return results

# Following function from Aneesha Bakharia
# https://aneesha.medium.com/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

def plot_coefficients(coefs, feature_names, current_class, top_features=10):
    """Plot the most important coefficients for a class."""
    plt.rcParams.update({'font.size': 30})
    top_positive_coefficients = np.argsort(coefs)[-top_features:]
    top_negative_coefficients = np.argsort(coefs)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    
    plt.figure(figsize=(15, 8))
    colors = ['red' if c < 0 else 'blue' for c in coefs[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coefs[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(0, 2 * top_features),
        feature_names[top_coefficients],
        rotation=60,
        ha='right',
        rotation_mode='anchor'
    )
    plt.title(f"Coefficients for {current_class}")
    plt.savefig(f'coefs_{current_class}.png', bbox_inches='tight')


def plot_rolling(final_predictions, smoothing=3, xlab="Index (segment center)"):
    """
    Plots the rolling stylometry results as lines of decision function values over the text.
    
    Parameters:
    final_predictions : Pandas dataframe containing the final predictions out of train_svm
        .
    
    smoothing : int or None
        The window size for smoothing the curves.
        - If smoothing is None or 0, no smoothing is applied.
        - If smoothing is an integer > 0, a simple moving average with that window size is applied.
        Default is 3, which provides a slight smoothing.
    """

    # Extract the segment center from the filename
    my_final_predictions = final_predictions.copy() # to avoid modifying in place
    segment_centers = []
    
    for fname in my_final_predictions['filename']:
        parts = fname.split('_')[-1].split('-')
        start = int(parts[0])
        end = int(parts[1])
        center = (start + end) / 2.0
        segment_centers.append(center)

    my_final_predictions['segment_center'] = segment_centers
    my_final_predictions['filename'] = [fname.split('_')[1] for fname in my_final_predictions['filename']]
    
    # Identify candidate columns
    known_cols = {'filename', 'author', 'segment_center'}
    candidate_cols = [c for c in my_final_predictions.columns if c not in known_cols]

    for work in my_final_predictions['filename'].unique():
        fpreds_work = my_final_predictions[my_final_predictions['filename'] == work]
        # Sort by segment center to ensure chronological order
        fpreds_work = fpreds_work.sort_values('segment_center')

        # Apply smoothing if requested
        if smoothing and smoothing > 0:
            for col in candidate_cols:
                fpreds_work[col] = fpreds_work[col].rolling(
                    window=smoothing, center=True, min_periods=1
                ).mean()

        # Plotting
        plt.figure(figsize=(24, 12))
        for col in candidate_cols:
            plt.plot(fpreds_work['segment_center'], fpreds_work[col], label=col, linewidth=2)

        plt.title(f'Rolling Stylometry Decision Functions Over {work}')
        plt.xlabel(xlab)
        plt.ylabel('Decision Function Value')
        plt.ylim(
            min(-2, min(fpreds_work[candidate_cols].min()) - 0.2),
            max(1, max(fpreds_work[candidate_cols].max())) + 0.2
        )
        plt.legend(title='Candidate Authors', fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'rolling_{work}.png', bbox_inches='tight')