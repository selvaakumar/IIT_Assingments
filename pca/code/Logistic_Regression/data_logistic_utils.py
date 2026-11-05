
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import pandas as pd
import matplotlib as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns



def plot_hist(data,y_label):
    import matplotlib.pyplot as plt
    X = data.drop(columns=[y_label], axis=1)
    features = X.columns

    fig, axes = plt.subplots(1, len(features), figsize=(15, 4.5))

    for ax, feat in zip(axes, features):
        sns.histplot(
            data=data,
            x=feat,
            hue=y_label,  # Optional: split by class
            bins=25,
            kde=True,
            palette=['#e74c3c', '#2ecc71'],
            element='step',
            stat='density',  # Normalizes counts when classes are imbalanced
            common_norm=False,
            ax=ax,
        )
    ax.set_title(f'Distribution of {feat}', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_cm(y_true, y_pred, labels=None, title="Confusion Matrix"):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(False)
    plt.tight_layout()
    
    # Crucial: trigger display
    plt.show()
    
    return fig, ax


def roc_curve_plot(y_true, y_scores):
    """
    Function to plot the ROC curve for a binary classification model.
    
    Parameters:
    y_true (array-like): True binary labels.
    y_scores (array-like): Target scores, can either be probability estimates of the positive class or confidence values.
    
    Returns:
    None
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def logitic_regression(X, y):
    """
    Function to perform logistic regression on the given dataset.
    
    Parameters:
    X (DataFrame): Features for the logistic regression model.
    y (Series): Target variable for the logistic regression model.
    
    Returns:
    model: Trained logistic regression model.
    """
    
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating the logistic regression model
    model = LogisticRegression()
    
    # Fitting the model to the training data
    model.fit(X_train, y_train)
    
    # Making predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluating the model
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    
    roc_curve_plot(y_test,y_pred)
    
    return model,y_test,y_pred, X_test

def data_exploration(data: pd.DataFrame) -> None:
    """Print key descriptive statistics and metadata of the dataset."""
    print("=" * 40)
    print("DATA EXPLORATION")
    print("=" * 40)
    print(f"Shape of dataset: {data.shape[0]} rows, {data.shape[1]} columns\n")
    print("Columns:", list(data.columns))
    print("\nData Types:\n", data.dtypes)
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nStatistical Summary:\n", data.describe(include="all"))
    print("=" * 40)
