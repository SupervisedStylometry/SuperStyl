# Class weights 
 
              precision    recall  f1-score   support

     BlonNes       0.15      0.12      0.14        16
       GaBru       0.35      0.32      0.34        34
    GautDarg       0.29      0.25      0.27        16
  GuilVinier       0.00      0.00      0.00        12
      Moniot       0.47      0.67      0.55        12
   ThibChamp       0.53      0.67      0.59        27

    accuracy                           0.37       117
   macro avg       0.30      0.34      0.31       117
weighted avg       0.33      0.37      0.35       117



     BlonNes       0.50      0.44      0.47        16
       GaBru       0.43      0.38      0.41        34
    GautDarg       0.33      0.31      0.32        16
   ThibChamp       0.56      0.70      0.62        27

    accuracy                           0.47        93
   macro avg       0.46      0.46      0.45        93
weighted avg       0.46      0.47      0.47        93

 
# Downsampling

## random downsampling without replacement

              precision    recall  f1-score   support

     BlonNes       0.27      0.33      0.30        12
       GaBru       0.46      0.50      0.48        12
    GautDarg       0.44      0.33      0.38        12
  GuilVinier       0.25      0.17      0.20        12
      Moniot       0.62      0.83      0.71        12
   ThibChamp       0.64      0.58      0.61        12

    accuracy                           0.46        72
   macro avg       0.45      0.46      0.45        72
weighted avg       0.45      0.46      0.45        72


At 16:

     BlonNes       0.64      0.44      0.52        16
       GaBru       0.25      0.25      0.25        16
    GautDarg       0.33      0.44      0.38        16
   ThibChamp       0.31      0.31      0.31        16

    accuracy                           0.36        64
   macro avg       0.38      0.36      0.36        64
weighted avg       0.38      0.36      0.36        64



## remove Tomek Links

              precision    recall  f1-score   support

     BlonNes       0.11      0.08      0.10        12
       GaBru       0.34      0.38      0.36        26
    GautDarg       0.29      0.17      0.21        12
  GuilVinier       0.00      0.00      0.00        12
      Moniot       0.43      0.60      0.50        10
   ThibChamp       0.52      0.71      0.60        21

    accuracy                           0.37        93
   macro avg       0.28      0.32      0.29        93
weighted avg       0.31      0.37      0.33        93

              precision    recall  f1-score   support

     BlonNes       0.43      0.38      0.40        16
       GaBru       0.39      0.42      0.41        26
    GautDarg       0.29      0.15      0.20        13
   ThibChamp       0.44      0.57      0.50        21

    accuracy                           0.41        76
   macro avg       0.39      0.38      0.38        76
weighted avg       0.40      0.41      0.40        76


## ENN -> vire trois tonnes de trucs !

              precision    recall  f1-score   support

  GuilVinier       0.86      1.00      0.92        12
   ThibChamp       1.00      0.33      0.50         3

    accuracy                           0.87        15
   macro avg       0.93      0.67      0.71        15
weighted avg       0.89      0.87      0.84        15

# Upsampling

## random upsampling with replacement 
 
              precision    recall  f1-score   support

     BlonNes       0.66      0.62      0.64        34
       GaBru       0.41      0.21      0.27        34
    GautDarg       0.68      0.79      0.73        34
  GuilVinier       0.82      0.82      0.82        34
      Moniot       0.85      1.00      0.92        34
   ThibChamp       0.66      0.79      0.72        34

    accuracy                           0.71       204
   macro avg       0.68      0.71      0.68       204
weighted avg       0.68      0.71      0.68       204

## SMOTE

              precision    recall  f1-score   support

     BlonNes       0.53      0.53      0.53        34
       GaBru       0.48      0.32      0.39        34
    GautDarg       0.71      0.65      0.68        34
  GuilVinier       0.82      0.79      0.81        34
      Moniot       0.79      1.00      0.88        34
   ThibChamp       0.65      0.76      0.70        34

    accuracy                           0.68       204
   macro avg       0.66      0.68      0.66       204
weighted avg       0.66      0.68      0.66       204

# Under- + Over-sampling

              precision    recall  f1-score   support

     BlonNes       0.47      0.45      0.46        31
       GaBru       0.40      0.27      0.32        30
    GautDarg       0.65      0.62      0.63        32
  GuilVinier       0.82      0.82      0.82        33
      Moniot       0.83      1.00      0.91        34
   ThibChamp       0.65      0.75      0.70        32

    accuracy                           0.66       192
   macro avg       0.63      0.65      0.64       192
weighted avg       0.64      0.66      0.65       192

###########################################################
######### SYSTEMATIC BENCHMARK WITH TRAIN/TEST SPLIT ######
###########################################################




class_weight
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC(class_weight='balanced'))]
              precision    recall  f1-score   support

     BlonNes       0.00      0.00      0.00         2
       GaBru       0.00      0.00      0.00         2
    GautDarg       0.00      0.00      0.00         2
  GuilVinier       0.00      0.00      0.00         2
      Moniot       0.25      0.50      0.33         2
   ThibChamp       0.33      0.50      0.40         2

    accuracy                           0.17        12
   macro avg       0.10      0.17      0.12        12
weighted avg       0.10      0.17      0.12        12

downsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
              precision    recall  f1-score   support

     BlonNes       0.00      0.00      0.00         2
       GaBru       0.00      0.00      0.00         2
    GautDarg       0.00      0.00      0.00         2
  GuilVinier       0.00      0.00      0.00         2
      Moniot       0.33      0.50      0.40         2
   ThibChamp       0.33      0.50      0.40         2

    accuracy                           0.17        12
   macro avg       0.11      0.17      0.13        12
weighted avg       0.11      0.17      0.13        12

Tomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
              precision    recall  f1-score   support

     BlonNes       0.00      0.00      0.00         2
       GaBru       0.33      0.50      0.40         2
    GautDarg       0.00      0.00      0.00         2
  GuilVinier       0.00      0.00      0.00         2
      Moniot       0.50      0.50      0.50         2
   ThibChamp       0.50      0.50      0.50         2

    accuracy                           0.25        12
   macro avg       0.22      0.25      0.23        12
weighted avg       0.22      0.25      0.23        12

upsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
              precision    recall  f1-score   support

     BlonNes       0.00      0.00      0.00         2
       GaBru       0.00      0.00      0.00         2
    GautDarg       0.00      0.00      0.00         2
  GuilVinier       0.25      0.50      0.33         2
      Moniot       0.50      0.50      0.50         2
   ThibChamp       0.33      0.50      0.40         2

    accuracy                           0.25        12
   macro avg       0.18      0.25      0.21        12
weighted avg       0.18      0.25      0.21        12

SMOTE
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
              precision    recall  f1-score   support

     BlonNes       0.00      0.00      0.00         2
       GaBru       0.50      0.50      0.50         2
    GautDarg       0.00      0.00      0.00         2
  GuilVinier       0.33      0.50      0.40         2
      Moniot       0.50      0.50      0.50         2
   ThibChamp       0.33      0.50      0.40         2

    accuracy                           0.33        12
   macro avg       0.28      0.33      0.30        12
weighted avg       0.28      0.33      0.30        12

SMOTETomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
              precision    recall  f1-score   support

     BlonNes       0.00      0.00      0.00         2
       GaBru       0.00      0.00      0.00         2
    GautDarg       0.00      0.00      0.00         2
  GuilVinier       0.25      0.50      0.33         2
      Moniot       0.50      0.50      0.50         2
   ThibChamp       0.33      0.50      0.40         2

    accuracy                           0.25        12
   macro avg       0.18      0.25      0.21        12
weighted avg       0.18      0.25      0.21        12


###### ON PSYCHÉ

##### BIG DATASET TESTED ON OUT-OF-DOMAIN


NO STRATEGY
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
               precision    recall  f1-score   support

        BOYER       1.00      1.00      1.00         9
   CORNEILLEP       0.82      1.00      0.90         9
   CORNEILLET       0.83      1.00      0.91        10
DONNEAUDEVISE       1.00      1.00      1.00         8
      MOLIERE       1.00      0.78      0.88         9
     QUINAULT       1.00      0.75      0.86         8

     accuracy                           0.92        53
    macro avg       0.94      0.92      0.92        53
 weighted avg       0.94      0.92      0.92        53

class_weight
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC(class_weight='balanced'))]
               precision    recall  f1-score   support

        BOYER       1.00      1.00      1.00         9
   CORNEILLEP       0.82      1.00      0.90         9
   CORNEILLET       0.83      1.00      0.91        10
DONNEAUDEVISE       1.00      1.00      1.00         8
      MOLIERE       1.00      0.78      0.88         9
     QUINAULT       1.00      0.75      0.86         8

     accuracy                           0.92        53
    macro avg       0.94      0.92      0.92        53
 weighted avg       0.94      0.92      0.92        53

downsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
               precision    recall  f1-score   support

        BOYER       1.00      1.00      1.00         9
   CORNEILLEP       0.82      1.00      0.90         9
   CORNEILLET       0.67      1.00      0.80        10
DONNEAUDEVISE       1.00      1.00      1.00         8
      MOLIERE       1.00      0.78      0.88         9
     QUINAULT       1.00      0.38      0.55         8

     accuracy                           0.87        53
    macro avg       0.91      0.86      0.85        53
 weighted avg       0.91      0.87      0.86        53

Tomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
               precision    recall  f1-score   support

        BOYER       1.00      1.00      1.00         9
   CORNEILLEP       0.82      1.00      0.90         9
   CORNEILLET       0.83      1.00      0.91        10
DONNEAUDEVISE       1.00      1.00      1.00         8
      MOLIERE       1.00      0.78      0.88         9
     QUINAULT       1.00      0.75      0.86         8

     accuracy                           0.92        53
    macro avg       0.94      0.92      0.92        53
 weighted avg       0.94      0.92      0.92        53

upsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
               precision    recall  f1-score   support

        BOYER       1.00      1.00      1.00         9
   CORNEILLEP       0.82      1.00      0.90         9
   CORNEILLET       0.83      1.00      0.91        10
DONNEAUDEVISE       1.00      1.00      1.00         8
      MOLIERE       1.00      0.67      0.80         9
     QUINAULT       1.00      0.88      0.93         8

     accuracy                           0.92        53
    macro avg       0.94      0.92      0.92        53
 weighted avg       0.94      0.92      0.92        53

SMOTE
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
               precision    recall  f1-score   support

        BOYER       1.00      1.00      1.00         9
   CORNEILLEP       0.82      1.00      0.90         9
   CORNEILLET       0.83      1.00      0.91        10
DONNEAUDEVISE       1.00      1.00      1.00         8
      MOLIERE       1.00      0.67      0.80         9
     QUINAULT       1.00      0.88      0.93         8

     accuracy                           0.92        53
    macro avg       0.94      0.92      0.92        53
 weighted avg       0.94      0.92      0.92        53

SMOTETomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC())]
               precision    recall  f1-score   support

        BOYER       1.00      1.00      1.00         9
   CORNEILLEP       0.82      1.00      0.90         9
   CORNEILLET       0.83      1.00      0.91        10
DONNEAUDEVISE       1.00      1.00      1.00         8
      MOLIERE       1.00      0.67      0.80         9
     QUINAULT       1.00      0.88      0.93         8

     accuracy                           0.92        53
    macro avg       0.94      0.92      0.92        53
 weighted avg       0.94      0.92      0.92        53


# LEAVE ONE OUT SAMPLING EVALUATION ON TROUVERES MELODIES
## NO SAMPLING


     BlonNes       0.15      0.12      0.14        16
       GaBru       0.39      0.47      0.43        34
    GautDarg       0.33      0.19      0.24        16
  GuilVinier       0.00      0.00      0.00        12
      Moniot       0.44      0.58      0.50        12
   ThibChamp       0.54      0.70      0.61        27

    accuracy                           0.40       117
   macro avg       0.31      0.35      0.32       117
weighted avg       0.35      0.40      0.37       117

class_weight
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC(class_weight='balanced'))]
.......... leave-one-out cross validation will be performed ........
.......... using 117 samples ........
              precision    recall  f1-score   support

     BlonNes       0.15      0.12      0.14        16
       GaBru       0.35      0.32      0.34        34
    GautDarg       0.29      0.25      0.27        16
  GuilVinier       0.00      0.00      0.00        12
      Moniot       0.47      0.67      0.55        12
   ThibChamp       0.53      0.67      0.59        27

    accuracy                           0.37       117
   macro avg       0.30      0.34      0.31       117
weighted avg       0.33      0.37      0.35       117

downsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', RandomUnderSampler(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 117 samples ........
              precision    recall  f1-score   support

     BlonNes       0.21      0.19      0.20        16
       GaBru       0.41      0.35      0.38        34
    GautDarg       0.29      0.31      0.30        16
  GuilVinier       0.29      0.33      0.31        12
      Moniot       0.50      0.75      0.60        12
   ThibChamp       0.68      0.63      0.65        27

    accuracy                           0.43       117
   macro avg       0.40      0.43      0.41       117
weighted avg       0.43      0.43      0.42       117

Tomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', TomekLinks()), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 117 samples ........
              precision    recall  f1-score   support

     BlonNes       0.00      0.00      0.00        16
       GaBru       0.42      0.44      0.43        34
    GautDarg       0.28      0.31      0.29        16
  GuilVinier       0.00      0.00      0.00        12
      Moniot       0.50      0.67      0.57        12
   ThibChamp       0.58      0.67      0.62        27

    accuracy                           0.39       117
   macro avg       0.30      0.35      0.32       117
weighted avg       0.34      0.39      0.37       117

upsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', RandomOverSampler(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 117 samples ........
              precision    recall  f1-score   support

     BlonNes       0.17      0.19      0.18        16
       GaBru       0.30      0.21      0.25        34
    GautDarg       0.40      0.50      0.44        16
  GuilVinier       0.00      0.00      0.00        12
      Moniot       0.53      0.67      0.59        12
   ThibChamp       0.53      0.63      0.58        27

    accuracy                           0.37       117
   macro avg       0.32      0.36      0.34       117
weighted avg       0.34      0.37      0.35       117

SMOTE
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', SMOTE(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 117 samples ........
              precision    recall  f1-score   support

     BlonNes       0.13      0.12      0.13        16
       GaBru       0.35      0.26      0.30        34
    GautDarg       0.28      0.31      0.29        16
  GuilVinier       0.11      0.08      0.10        12
      Moniot       0.53      0.75      0.62        12
   ThibChamp       0.56      0.67      0.61        27

    accuracy                           0.38       117
   macro avg       0.33      0.37      0.34       117
weighted avg       0.35      0.38      0.36       117

SMOTETomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', SMOTETomek(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 117 samples ........
              precision    recall  f1-score   support

     BlonNes       0.12      0.12      0.12        16
       GaBru       0.48      0.35      0.41        34
    GautDarg       0.24      0.31      0.27        16
  GuilVinier       0.09      0.08      0.09        12
      Moniot       0.50      0.67      0.57        12
   ThibChamp       0.61      0.63      0.62        27

    accuracy                           0.38       117
   macro avg       0.34      0.36      0.35       117
weighted avg       0.39      0.38      0.38       117

# On trouveres texts

# LEAVE ONE OUT SAMPLING EVALUATION ON TROUVERES TEXT
## NO SAMPLING

              precision    recall  f1-score   support

       Blond       0.54      0.33      0.41        21
        Gace       0.61      0.88      0.72        43
        Gaut       0.64      0.35      0.45        20
        Thib       0.98      0.93      0.95        45

    accuracy                           0.73       129
   macro avg       0.69      0.63      0.64       129
weighted avg       0.73      0.73      0.71       129


# class_weight
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC(class_weight='balanced'))]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.60      0.43      0.50        21
        Gace       0.67      0.84      0.74        43
        Gaut       0.59      0.50      0.54        20
        Thib       0.98      0.93      0.95        45

    accuracy                           0.75       129
   macro avg       0.71      0.67      0.68       129
weighted avg       0.75      0.75      0.75       129

# downsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', RandomUnderSampler(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.52      0.52      0.52        21
        Gace       0.81      0.58      0.68        43
        Gaut       0.47      0.80      0.59        20
        Thib       0.98      0.93      0.95        45

    accuracy                           0.73       129
   macro avg       0.69      0.71      0.69       129
weighted avg       0.77      0.73      0.74       129

# Tomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', TomekLinks()), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.75      0.57      0.65        21
        Gace       0.66      0.86      0.75        43
        Gaut       0.75      0.45      0.56        20
        Thib       0.93      0.93      0.93        45

    accuracy                           0.78       129
   macro avg       0.77      0.70      0.72       129
weighted avg       0.78      0.78      0.77       129

# upsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', RandomOverSampler(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.60      0.43      0.50        21
        Gace       0.69      0.84      0.76        43
        Gaut       0.63      0.60      0.62        20
        Thib       0.98      0.93      0.95        45

    accuracy                           0.77       129
   macro avg       0.73      0.70      0.71       129
weighted avg       0.77      0.77      0.76       129

# SMOTE
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', SMOTE(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.60      0.43      0.50        21
        Gace       0.71      0.84      0.77        43
        Gaut       0.65      0.65      0.65        20
        Thib       0.98      0.93      0.95        45

    accuracy                           0.78       129
   macro avg       0.73      0.71      0.72       129
weighted avg       0.77      0.78      0.77       129

# SMOTETomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', SMOTETomek(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.60      0.43      0.50        21
        Gace       0.71      0.84      0.77        43
        Gaut       0.65      0.65      0.65        20
        Thib       0.98      0.93      0.95        45

    accuracy                           0.78       129
   macro avg       0.73      0.71      0.72       129
weighted avg       0.77      0.78      0.77       129

# LEAVE ONE OUT SAMPLING EVALUATION ON TROUVERES TEXT
## NO SAMPLING
# class_weight
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('model', LinearSVC(class_weight='balanced'))]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.64      0.43      0.51        21
        Gace       0.70      0.81      0.75        43
        Gaut       0.68      0.65      0.67        20
        Thib       0.91      0.93      0.92        45

    accuracy                           0.77       129
   macro avg       0.74      0.71      0.71       129
weighted avg       0.76      0.77      0.76       129

# downsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', RandomUnderSampler(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.42      0.48      0.44        21
        Gace       0.74      0.53      0.62        43
        Gaut       0.48      0.75      0.59        20
        Thib       0.98      0.93      0.95        45

    accuracy                           0.70       129
   macro avg       0.65      0.67      0.65       129
weighted avg       0.73      0.70      0.70       129

# Tomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', TomekLinks()), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.85      0.52      0.65        21
        Gace       0.69      0.86      0.76        43
        Gaut       0.71      0.60      0.65        20
        Thib       0.93      0.93      0.93        45

    accuracy                           0.79       129
   macro avg       0.79      0.73      0.75       129
weighted avg       0.80      0.79      0.79       129

# upsampling
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', RandomOverSampler(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.64      0.43      0.51        21
        Gace       0.70      0.81      0.75        43
        Gaut       0.68      0.65      0.67        20
        Thib       0.91      0.93      0.92        45

    accuracy                           0.77       129
   macro avg       0.74      0.71      0.71       129
weighted avg       0.76      0.77      0.76       129

# SMOTE
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', SMOTE(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.64      0.43      0.51        21
        Gace       0.71      0.79      0.75        43
        Gaut       0.67      0.70      0.68        20
        Thib       0.91      0.93      0.92        45

    accuracy                           0.77       129
   macro avg       0.73      0.71      0.72       129
weighted avg       0.76      0.77      0.76       129

# SMOTETomek
.......... loading data ........
.......... Formatting data ........
.......... Creating pipeline according to user choices ........
.......... using normalisations ........
.......... implementing strategy to solve imbalance in data ........
.......... choosing SVM ........
.......... Creating pipeline with steps ........
[('scaler', StandardScaler()), ('normalizer', Normalizer()), ('sampling', SMOTETomek(random_state=42)), ('model', LinearSVC())]
.......... leave-one-out cross validation will be performed ........
.......... using 129 samples ........
              precision    recall  f1-score   support

       Blond       0.64      0.43      0.51        21
        Gace       0.71      0.79      0.75        43
        Gaut       0.67      0.70      0.68        20
        Thib       0.91      0.93      0.92        45

    accuracy                           0.77       129
   macro avg       0.73      0.71      0.72       129
weighted avg       0.76      0.77      0.76       129

