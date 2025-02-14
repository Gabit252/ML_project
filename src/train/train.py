import argparse
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml")
    return parser.parse_args()

def train_model(config):
    processed_path = "data/processed/processed.csv"
    df = pd.read_csv(processed_path)
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'] 
    
    test_size = config['train']['test_size']
    random_state = config['train']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model_params = config['train']['model_params']
    n_components = config['train']['n_components']
    
    pipepline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('logreg', LogisticRegression(**model_params)) 
])
    
    grid_params = {'logreg__C': np.logspace(-3, 3, 7), 'logreg__penalty': ['l1','l2','elasticnet']}
    Grid = GridSearchCV(pipepline, grid_params, cv=5, scoring='accuracy')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    loadings = pd.DataFrame(
    pca.components_.T,
    columns = ['PC1','PC2'],
    index = X.columns)
    
    mlflow.autolog()
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run():
        Grid.fit(X_train, y_train)
        preds = Grid.predict(X_test)
        pred_proba = Grid.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        auc_score = auc(fpr, tpr)

        #Log AUC score
        #mlflow.log_metric("AUC", auc_score)

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Save the figure
        roc_curve_path = "artifacts/roc_curve.png"
        plt.savefig(roc_curve_path)
        plt.close()

        # Log ROC curve image in MLflow
        #mlflow.log_artifact(roc_curve_path)
        
        cm = confusion_matrix(y_test, preds)

        cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        labels = np.asarray([
            [f'TN\n{cm[0,0]}\n({cm_percentages[0,0]:.1f}%)', f'FP\n{cm[0,1]}\n({cm_percentages[0,1]:.1f}%)'],
            [f'FN\n{cm[1,0]}\n({cm_percentages[1,0]:.1f}%)', f'TP\n{cm[1,1]}\n({cm_percentages[1,1]:.1f}%)']
        ])

        plt.figure(figsize=(10,8))
        sns.heatmap(cm, fmt='', annot=labels, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.gca().set_xticklabels(['Negative (0)', 'Positive (1)'])
        plt.gca().set_yticklabels(['Negative (0)', 'Positive (1)'])
        plt.tight_layout()
        
        confusion_matrix_png = "artifacts/confusion_matrix.png"
        plt.savefig(confusion_matrix_png)
        plt.close()
        
        #mlflow.log_artifact(confusion_matrix_png)
        
        # Визуализируем данные в пространстве главных компонент
        plt.figure(figsize=(8,6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap = 'rainbow', edgecolors='k',s = 80, alpha=0.8)
        plt.xlabel('Первая главная компонента (PC1)')
        plt.ylabel('Вторая главная компонента (PC2)')
        plt.title('Проекция данных на две главные компоненты', fontsize = 14)
        plt.grid(True, linestyle = '--', alpha = 0.7)
        
        PCA_png = "artifacts/PCA.png"
        plt.savefig(PCA_png)
        plt.close()
        
       #mlflow.log_artifact(PCA_png)
        
        # Визуализация вклада признаков в главные компоненты с помощью тепловой карты
        plt.figure(figsize=(10,6))
        sns.heatmap(loadings, annot=True, cmap = 'coolwarm', center = 0) 
        plt.title('Вклад признаков в главные компоненты', fontsize=14)
        plt.tight_layout()
        

        PCA_heatmap_png = "artifacts/PCA_heatmap.png"
        plt.savefig(PCA_heatmap_png)
        plt.close()
        
        #mlflow.log_artifact(PCA_heatmap_png)

        #mlflow.log_params(model_params)
        #mlflow.log_param("test_size", test_size)
        #mlflow.log_metric("accuracy", acc)
        #mlflow.log_metric("precision", prec)
        
        
                
        model_path = "models/model.pkl"
        joblib.dump(Grid, model_path)
        #mlflow.sklearn.log_model(Grid, "model")
        print(f"Model trained with accuracy: {acc}")
        


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_model(config)

if __name__ == "__main__":
    main()