import os
import time
import joblib
import curlify
import warnings
import datetime
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union

import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from scipy.stats import mannwhitneyu

from sklearn.tree import plot_tree
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score,\
     classification_report, confusion_matrix, precision_recall_curve, auc

import config as c

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

tic = time.time()


class WineSelector:
    """
    One-stop shop for all analytics and ML
    """

    def __init__(self):
        self.target_variable = c.TARGET_VARIABLE
        self.file_path = c.FILE_PATH

    def data_loader(self, mode: str='package', file_path=None) -> pd.DataFrame:
        """
        Takes the data either from the package or from csv and loads it in pandas df

        :param mode: package to download the data; file to read a csv. In this case file_path should be given
        :param file_path: full path with file name
        :return: data for red wine (will filter out white wine in case of package)
        """

        if mode == 'file':
            if file_path is None:
                raise ValueError("File path must be provided.")

            data = pd.read_csv(file_path, sep=';')

        else:
            raw_data = fetch_ucirepo(id=186)

            data = pd.DataFrame(raw_data.data.original)
            data['quality'] = raw_data.data.targets
            data = data.query("color == 'red'")

        return data

    def data_exploration(self, df: pd.DataFrame, plt_subfolder=None) -> None:
        """
        Performance various analysis and visualizations - Target and Feature distribution, correlation plot,
        box-plot, violin plot...

        :param df: data frame with all features and target variable
        :param plt_subfolder: folder name to store the graphs (if applicable)
        :return:
        """
        if not os.path.exists(f'graphs'):
            os.makedirs(f'graphs')

        if not os.path.exists(f'graphs/{plt_subfolder}'):
            os.makedirs(f'graphs/{plt_subfolder}')

        plt.ioff()  # Don't show the graphs

        print("Dataset Info:")
        print(df.info())

        print("\nSummary Statistics:")
        print(df.describe())

        # Visualize the distribution of the target variable
        percentage = df['quality'].value_counts(normalize=True) * 100
        percentage.sort_index(inplace=True)
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(x='quality', data=df)
        for p, label in zip(ax.patches, percentage):
            ax.annotate(f'{label:.0f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                        xytext=(0, 7), textcoords='offset points')

        plt.title("Distribution of Wine Quality Scores (target)")
        plt.savefig(f'graphs/{plt_subfolder}/target_distribution.jpg', dpi=150)
        # plt.show()
        plt.close('all')

        # Visualize the correlation matrix for numeric features
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'size':8})
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f'graphs/correlation_matrix.jpg', dpi=150)
        # plt.show()
        plt.close('all')

        # Boxplot to identify outliers in numeric features
        for column in df.columns[:-1]:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='quality', y=column, data=df)
            plt.ylabel(column)
            plt.title(f'Boxplot for {column}', fontsize=18)
            plt.ioff()
            plt.savefig(f'graphs/{plt_subfolder}/box_plot_{column}.jpg', dpi=150)
            # plt.show()

        # Violin plot for feature distribution
        for column in df.columns[:-1]:
            plt.figure(figsize=(12, 8))
            sns.violinplot(x='quality', y=column, data=df)
            plt.title(f"Violin Plot of {column} by {self.target_variable}", fontsize=22)
            plt.ioff()
            plt.savefig(f'graphs/{plt_subfolder}/violin_plot_{column}.jpg', dpi=150)
            # plt.show()
            plt.close('all')

        # df[self.target_variable] = df[self.target_variable].astype('category')
        # Create bins based on feature variable distribution
        for column in df.columns[:-1]:

            bins = np.linspace(df[column].min(), df[column].max(), 10 + 1)
            rounded_bins = bins[:-1]
            df['bin'] = pd.cut(df[column], bins=bins, labels=rounded_bins, include_lowest=True)
            counts = df.groupby('bin')[self.target_variable].value_counts().unstack().fillna(0)
            ax = counts.plot(kind='bar', stacked=True, figsize=(10, 6), width=0.8)
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.title(f'Histogram with Cutoffs for {column}', fontsize=18)
            plt.legend(title=self.target_variable)
            plt.xticks(range(len(rounded_bins)), [f'{x:.2f}' for x in rounded_bins], rotation=45)

            plt.ioff()
            # plt.show()
            plt.savefig(f'graphs/{plt_subfolder}/histogram_{column}.jpg', dpi=150)
            df.drop('bin', axis=1, inplace=True)

    def remove_outliers_iqr(self, data: pd.DataFrame, threshold: int=1.5) -> pd.Series:
        """
        IQR test to flag and remove outliers.
        It returns an outlier mask (doesn't directly remove the outliers from the data),
        so the output can be applied to multiple dfs

        :param data: The features for which outliers would be flagged. It can be only subset of features
        :param threshold: The IQR multiplier. Default is the standard 1.5
        :return: Outlier mask/filter
        """
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers_mask = ((data < lower_bound) | (data > upper_bound)).any(axis=1)

        return outliers_mask

    def data_preparation(self, df: pd.DataFrame, features_for_iqr: Union[str, list] = 'all') -> None:
        """
        Generates 3 traimning/testing data sets
        - Original: data as is
        - No outliers: based on the IQR mask
        - Scaled: for the Logistic regression

        :param df: Dataset with features and target
        :param features_for_iqr: which features would be considered for outlier removal
        :return: Creates the datasets within the class
        """

        y = df[self.target_variable]
        X = df.drop(self.target_variable, axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=17)

        if features_for_iqr == 'all':
            outliers_mask = self.remove_outliers_iqr(X)
        else:
            outliers_mask = self.remove_outliers_iqr(X[features_for_iqr])

        self.X_train_cleaned = self.X_train[~outliers_mask]
        self.y_train_cleaned = self.y_train[~outliers_mask]

        # Standardize the features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train_cleaned)
        self.y_train_scaled = self.y_train[~outliers_mask]
        self.X_test_scaled = scaler.transform(self.X_test)

    def grid_search(self, models: list, param_distributions: dict, labels: list) -> pd.DataFrame:
        """
        Performs GridSearchCV for multiple models and store the best models along with performance metrics.
        Note: The optimization is done using Macro precision
        Parameters:
        :param models: List of tuples, each containing a model name and its corresponding sklearn model class.
        :param param_distributions: Dictionary, parameter distributions for GridSearchCV for each model.
        :param X_train, y_train: Training data and labels.
        :param X_test, y_test: Testing data and labels.
        :param labels: List of values for the target variable
        :return: DataFrame, containing the best models of each type/dataset and their metrics and parameters
        """
        best_models = pd.DataFrame()
        for data_set in range(3):

            if data_set == 0:
                X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test
            elif data_set == 1:
                X_train, y_train, X_test, y_test = self.X_train_cleaned, self.y_train_cleaned, self.X_test, self.y_test
            elif data_set == 2:
                X_train, y_train, X_test, y_test = self.X_train_scaled, self.y_train_scaled, self.X_test_scaled, self.y_test

            for model_name, model_class in models:
                print(f"\nPerforming GridSearchCV for {model_name} and data set {data_set}...")

                # Perform Grid Search
                model = model_class()
                precision_scorer = make_scorer(precision_score, average='macro')

                warnings.simplefilter("ignore")
                grid_search = GridSearchCV(estimator=model, param_grid=param_distributions[model_name],
                                             scoring=precision_scorer, cv=5, verbose=2)
                warnings.resetwarnings()
                # random_search.fit(X_resampled, y_resampled)
                grid_search.fit(X_train, y_train)
                cv_results = grid_search.cv_results_
                # Get the best model
                best_model = grid_search.best_estimator_

                # Make predictions on the test set
                y_pred = best_model.predict(X_test)

                metrics = classification_report(y_test, y_pred, target_names=labels,
                                                output_dict=True)
                metrics = pd.DataFrame(metrics).transpose().reset_index()
                metrics['model_name'] = model_name
                metrics['data_set'] = data_set
                metrics['parameters'] = str(grid_search.best_params_)
                # Store the best model and precision score
                # best_models[model_name] = {'model': best_model, 'precision': precision}

                best_models = pd.concat([best_models, metrics], ignore_index=True)

                print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
                print(f"Precision for {model_name} Class 1: {metrics[metrics['index'] == 1]['precision'].iloc[0]}")
            print('Done!')

        best_models.to_csv('best_models_metrics.csv', index=False)

        return best_models

    def plot_precision_recall_curve(self, y_true: list, y_scores: list,
                                    threshold_points: list = None, title='Precision-Recall Curve') -> None:
        """
        Plot the precision-recall curve for a binary classification model.

        :param y_true: True binary labels
        :param y_scores: Predicted probabilities or decision function scores
        :param title: Title for the plot (default is 'Precision-Recall Curve')
        :return: None
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        area_under_curve = auc(recall, precision)
        plt.figure(figsize=(20, 10))

        plt.plot(recall, precision, label=f'AUC = {area_under_curve:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)

        # Plot threshold points with labels
        if threshold_points:
            for threshold_point in threshold_points:
                idx = next(idx for idx, val in enumerate(thresholds) if val >= threshold_point)
                plt.scatter(recall[idx], precision[idx], marker='o', color='red',
                            label=f'Threshold: {threshold_point:.2f}')
                plt.annotate(f'{threshold_point:.2f}', xy=(recall[idx], precision[idx]),
                             xytext=(recall[idx] - 0.1, precision[idx] + 0.1),
                             arrowprops=dict(facecolor='black', arrowstyle='->'))
        # plt.show()
        plt.savefig(f'graphs/precision_recall_curve.jpg', dpi=300, format='jpg')
        plt.close('all')

    def plot_tree_model_importance(self, model, X, feature_names=None) -> None:
        """
        Horizontal bar blot with the sorted importance of the model's features

        :param model: Trained tree based model - e.g. Random Forest
        :param X: Training set
        :param feature_names: List of feature names
        :return: Saves a graph with the model's feature importance
        """

        feature_importances = model.feature_importances_

        # Use provided feature names or default to indexes
        if feature_names is None:
            feature_names = np.arange(X.shape[1])

        sorted_indices = np.argsort(feature_importances)

        plt.figure(figsize=(15, 10))
        bars = plt.barh(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
        plt.yticks(range(len(feature_importances)), feature_names[sorted_indices])
        plt.xlabel('Feature Importance (%)')
        plt.title('Random Forest Feature Importances')

        for bar, percent in zip(bars, feature_importances[sorted_indices] * 100):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f'{percent:.0f}%', va='center', color='black', fontsize=14)
        plt.savefig(f'graphs/model_feature_importance.jpg', dpi=300, format='jpg')
        # plt.show()
        plt.close('all')

    def visualize_decision_tree(self, tree_model, feature_names) -> None:
        """
        Visualize a decision tree.

        :param tree_model: Trained decision tree model
        :param feature_names: List of feature names
        :return: Saves a graph with the Decision tree visualization
        """
        plt.figure(figsize=(20, 10))
        plot_tree(tree_model, feature_names=feature_names, class_names=['not high', 'high'], filled=True, rounded=True)
        # plt.show()
        plt.savefig(f'graphs/decision_tree.jpg', dpi=1300, format='jpg')

    def mannwhitneyu_test(self, df: pd.DataFrame) -> None:
        """
        Performs Mann-Whitneyu test and prints results

        :param df: Dataset with the features and target variable
        :return: Prints the results
        """

        group_0 = df[df[self.target_variable] == 0]
        group_1 = df[df[self.target_variable] == 1]

        for feature in df.columns[:-1]:
            stat, p_value = mannwhitneyu(group_0[feature], group_1[feature], alternative='two-sided')
            print(f"Mann-Whitney U test for {feature}: U-statistic = {stat}, p-value = {p_value, 4}")

if __name__ == '__main__':

    # -- Load the data from file
    red_wine = WineSelector()
    rw_data = red_wine.data_loader(mode='file', file_path=c.FILE_PATH)

    # -- First step of EDA - consider the original distribution of the target variable
    red_wine.data_exploration(rw_data, 'orig')

    # -- Second step of EDA - consider only 2 classes for quality
    df_two_classes = rw_data.copy()
    df_two_classes['quality'] = np.where(rw_data['quality'] < 7, 0, 1)
    red_wine.data_exploration(df_two_classes, 'bucket')

    # -- Non-parametric test for difference in distributions
    red_wine.mannwhitneyu_test(df_two_classes)

    # Tested 3 classes, but it doesn't work and ultimately would not add much to the use-case specs
    # df_two_classes['quality'] = np.where((df['quality'] >= 4) & (df['quality'] < 7), 1, df_two_classes['quality1'])
    # df_two_classes.drop('quality1', axis=1, inplace=True)

    # -- Prepare the training/test datasets
    red_wine.data_preparation(df_two_classes, c.CATEGORIES_FOR_OUTLIERS)

    # -- Grid search to select the best performing model with optimization goal macro precision

    # Sample grid search for testing
    # models = [('Logistic Regression', LogisticRegression)]
    # param_distributions = {
    #     'Logistic Regression': {'random_state': [17],
    #                             'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                             'penalty': [None, 'l1', 'l2', 'elasticnet'],
    #                             'class_weight': [None, 'balanced', {0: 1, 1: 10}]}}

    # model_summary = red_wine.grid_search(models, param_distributions, [0, 1])

    models = c.MODELS
    param_distributions = c.PARAM_DISTRIBUTION

    model_summary = red_wine.grid_search(models, param_distributions, [0, 1])  # -- Re-run if needed
    # model_summary = pd.read_csv('best_models_metrics.csv')
    best_model_idx = 5
    best_model_class = model_summary['model_name'].iloc[best_model_idx].lower().replace(' ', '_')

    #-- Recommendation 1: Preferred option - Train the best performing model
    wine_model = RandomForestClassifier(**eval(model_summary['parameters'].iloc[best_model_idx]))
    wine_model.fit(red_wine.X_train, red_wine.y_train)
    ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    joblib.dump(wine_model, f"model/high_quality_red_wine_classifier_{best_model_class}_{ts}.pkl")

    y_pred = wine_model.predict(red_wine.X_test)
    print(confusion_matrix(red_wine.y_test, y_pred))
    print(classification_report(red_wine.y_test, y_pred))

    # -- Precision-Recall curve to identify the optimal cut-off for a scenarios
    # 1) When new wines would be introduced
    # 2) Current assortment needs to be scored
    y_scores = wine_model.predict_proba(red_wine.X_test)[:, 1]
    threshold_points = [0.4, 0.5, 0.65]
    red_wine.plot_precision_recall_curve(red_wine.y_test, y_scores, threshold_points)
    red_wine.plot_tree_model_importance(wine_model, red_wine.X_train, red_wine.X_train.columns)

    # -- Recommendation 2: Fallout option - Rule of thumb - explore the decision tree and define recipe for good wine
    tree_model = DecisionTreeClassifier(max_depth=3)
    tree_model.fit(red_wine.X_train, red_wine.y_train)
    red_wine.visualize_decision_tree(tree_model, red_wine.X_train.columns)

    # -- Test the API
    url = 'http://127.0.0.1:5002/predict'  # Local testing
    input_data = {'features': {'fixed acidity': 7.2,
                               'volatile acidity': 0.38,
                               'citric acid': 0.38,
                               'residual sugar': 2.8,
                               'chlorides': 0.068,
                               'free sulfur dioxide': 10.00,
                               'total sulfur dioxide': 42.00,
                               'density': 0.99,
                               'pH': 3.34,
                               'sulphates': 0.72,
                               'alcohol': 12.9}
                  }
    response = requests.post(url, json=input_data)
    print(curlify.to_curl(response.request))

    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']
        print(f"Prediction: {prediction}")
    else:
        print(f"Error: {response.text}")

    toc = time.time()
    print(f'Done in {toc - tic}')
