import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mtick
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

class GenderPredictor:
    def __init__(self):
        self.model = None
        self.product_categories = {}
        self.product_counts = {}
        self.category_popularity = {}
        self.visualizations_path = "visualizations/"
        # Create visualization directory if it doesn't exist
        import os
        if not os.path.exists(self.visualizations_path):
            os.makedirs(self.visualizations_path)
    
    def extract_product_info(self, product_id_str):
        """Extract detailed product information from the product ID string"""
        if pd.isna(product_id_str) or product_id_str == '':
            return []
        
        products = []
        for item in str(product_id_str).strip('/').split('/'):
            if item and re.match(r'^[A-Z]\d+$', item):
                products.append(item)
        
        return products
    
    def extract_advanced_features(self, df):
        """
        Advanced feature extraction from raw data
        """
        # Convert time columns
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Session duration
        df['session_duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
        
        # Time-based features
        df['start_hour'] = df['start_time'].dt.hour
        df['is_weekend'] = df['start_time'].dt.dayofweek >= 5
        df['day_of_week'] = df['start_time'].dt.dayofweek
        df['part_of_day'] = pd.cut(df['start_time'].dt.hour, 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['night', 'morning', 'afternoon', 'evening'])
        
        # Product analysis
        df['total_products'] = df['product_ids'].apply(lambda x: len(self.extract_product_info(x)))
        
        # Extract categories from product IDs
        df['product_cats'] = df['product_ids'].apply(
            lambda x: ' '.join([p for p in self.extract_product_info(x)]) if pd.notna(x) else ''
        )
        
        # Debug print to check the content
        print("Sample of product_cats:", df['product_cats'].head())
        print("Number of empty product_cats:", df['product_cats'].isna().sum())
        
        # Parse product category structure
        for _, row in df.iterrows():
            products = self.extract_product_info(row['product_ids'])
            
            for product in products:
                if product not in self.product_counts:
                    self.product_counts[product] = 0
                self.product_counts[product] += 1
                
                # Store category info
                cat = product[:1]  # First letter is category
                if cat not in self.product_categories:
                    self.product_categories[cat] = 0
                self.product_categories[cat] += 1
        
        # Calculate category popularity (normalize counts)
        total = sum(self.product_categories.values())
        for cat, count in self.product_categories.items():
            self.category_popularity[cat] = count / total
        
        # Add category popularity features
        for cat, pop in self.category_popularity.items():
            col_name = f'cat_{cat}_ratio'
            df[col_name] = df['product_cats'].apply(
                lambda x: x.count(cat) / len(x) if len(x) > 0 else 0
            )
        
        # Behavioral features
        df['unique_categories'] = df['product_cats'].apply(lambda x: len(set(x.split())))
        df['category_diversity'] = df['unique_categories'] / df['total_products'].apply(lambda x: max(x, 1))
        
        return df

    def custom_score(self, y_true, y_pred):
        """Calculate the custom gender balance score"""
        male_mask = y_true == 'male'
        female_mask = y_true == 'female'
        
        male_total = np.sum(male_mask)
        female_total = np.sum(female_mask)
        
        if male_total == 0 or female_total == 0:
            return 0
        
        male_correct = np.sum((male_mask) & (y_pred == 'male'))
        female_correct = np.sum((female_mask) & (y_pred == 'female'))
        
        return (male_correct / male_total + female_correct / female_total) / 2

    def visualize_data_distribution(self, df):
        """Visualize the distribution of data features"""
        # First extract time-based features
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Create all necessary features
        df['start_hour'] = df['start_time'].dt.hour
        df['day_of_week'] = df['start_time'].dt.dayofweek
        df['session_duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
        df['part_of_day'] = pd.cut(df['start_time'].dt.hour, 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'])
        
        # Calculate product-related features
        df['total_products'] = df['product_ids'].apply(lambda x: len(self.extract_product_info(x)))
        
        # Create product_cats first
        df['product_cats'] = df['product_ids'].apply(
            lambda x: ' '.join([p for p in self.extract_product_info(x)]) if pd.notna(x) else ''
        )
        
        # Now we can calculate unique_categories
        df['unique_categories'] = df['product_cats'].apply(lambda x: len(set(x.split())))
        df['category_diversity'] = df['unique_categories'] / df['total_products'].apply(lambda x: max(x, 1))
        
        # Now create the visualizations
        plt.figure(figsize=(10, 6))
        sns.countplot(x='gender', data=df)
        plt.title('Gender Distribution in Dataset')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.savefig(f"{self.visualizations_path}gender_distribution.png")
        plt.close()
        
        # Visualize time-based patterns
        plt.figure(figsize=(15, 10))
        
        # Hour of day distribution by gender
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='start_hour', hue='gender', multiple='stack', bins=24)
        plt.title('Shopping Time Distribution by Gender')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count')
        
        # Day of week distribution
        plt.subplot(2, 2, 2)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_day = df.copy()
        df_day['day_name'] = df_day['day_of_week'].apply(lambda x: days[x])
        sns.countplot(x='day_name', hue='gender', data=df_day)
        plt.title('Shopping Day Distribution by Gender')
        plt.xlabel('Day of Week')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # Part of day distribution
        plt.subplot(2, 2, 3)
        sns.countplot(x='part_of_day', hue='gender', data=df)
        plt.title('Part of Day Distribution by Gender')
        plt.xlabel('Part of Day')
        plt.ylabel('Count')
        
        # Session duration distribution
        plt.subplot(2, 2, 4)
        sns.histplot(data=df, x='session_duration', hue='gender', multiple='stack', bins=30)
        plt.title('Session Duration by Gender')
        plt.xlabel('Session Duration (seconds)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{self.visualizations_path}time_patterns.png")
        plt.close()
        
        # Visualize product category preferences by gender
        plt.figure(figsize=(14, 8))
        
        # Category diversity
        plt.subplot(1, 2, 1)
        sns.boxplot(x='gender', y='category_diversity', data=df)
        plt.title('Category Diversity by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Category Diversity')
        
        # Total products viewed
        plt.subplot(1, 2, 2)
        sns.boxplot(x='gender', y='total_products', data=df)
        plt.title('Total Products Viewed by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Total Products')
        
        plt.tight_layout()
        plt.savefig(f"{self.visualizations_path}shopping_behavior.png")
        plt.close()
        
        # Product category preferences
        cat_cols = [col for col in df.columns if col.startswith('cat_') and col.endswith('_ratio')]
        if cat_cols:
            plt.figure(figsize=(16, 8))
            
            # Create a dataframe of category ratios by gender
            cat_data = df.groupby('gender')[cat_cols].mean().reset_index()
            cat_data_melted = pd.melt(cat_data, id_vars=['gender'], value_vars=cat_cols, 
                                    var_name='Category', value_name='Ratio')
            cat_data_melted['Category'] = cat_data_melted['Category'].str.replace('cat_', '').str.replace('_ratio', '')
            
            # Plot category preferences
            sns.barplot(x='Category', y='Ratio', hue='gender', data=cat_data_melted)
            plt.title('Product Category Preferences by Gender')
            plt.xlabel('Product Category')
            plt.ylabel('Average Ratio in Shopping Session')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.visualizations_path}category_preferences.png")
            plt.close()
    
    def visualize_feature_importance(self):
        """I'm creating a visualization of feature importance from my trained model"""
        if self.model is None:
            print("I haven't trained the model yet, so I can't visualize feature importance.")
            return
        
        # I'll store my feature names in this list
        feature_names = []
        
        # I'm going to try to extract feature names from each transformer
        for name, trans, cols in self.preprocessor.transformers_:
            if name == 'num':
                # For numeric features, I'll just use their original names
                feature_names.extend(cols)
            elif name == 'cat_part_day':
                # I need to create names for each part of day category
                feature_names.extend([f'part_of_day_{label}' for label in ['night', 'morning', 'afternoon', 'evening']])
            elif name == 'cat_weekend':
                # Weekend is a simple boolean feature
                feature_names.extend(['is_weekend'])
            elif name == 'cat_products':
                # For my TF-IDF features, I'll create generic numbered names
                n_features = trans.named_steps['tfidf'].get_feature_names_out().shape[0]
                feature_names.extend([f'product_feature_{i}' for i in range(n_features)])
        
        # I should make sure I have the right number of feature names
        if len(feature_names) != len(self.model.feature_importances_):
            print(f"Oops, I've got a mismatch in feature names. I expected {len(self.model.feature_importances_)}, but got {len(feature_names)}")
            feature_names = [f"Feature {i}" for i in range(len(self.model.feature_importances_))]
        
        # Now I'll create a dataframe to hold my feature importance data
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # I'll plot the top 20 most important features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(20))
        plt.title('My Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(f"{self.visualizations_path}feature_importance.png")
        plt.close()
        
        # I'll return the dataframe in case I want to analyze it further
        return fi_df
    
    def visualize_model_performance(self, y_test, y_pred, y_prob=None):
        """Visualize model performance metrics"""
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Female', 'Male'], 
                    yticklabels=['Female', 'Male'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f"{self.visualizations_path}confusion_matrix.png")
        plt.close()
        
        # ROC Curve (if probability predictions are available)
        if y_prob is not None:
            # For binary classification, we need the probability of the positive class
            # Assuming 'male' is the positive class (index 1)
            if y_prob.shape[1] >= 2:
                y_prob_positive = y_prob[:, 1]
                fpr, tpr, _ = roc_curve(y_test == 'male', y_prob_positive)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.visualizations_path}roc_curve.png")
                plt.close()
        
        # Class-wise accuracy
        male_mask = y_test == 'male'
        female_mask = y_test == 'female'
        
        male_correct = np.sum((male_mask) & (y_pred == 'male'))
        female_correct = np.sum((female_mask) & (y_pred == 'female'))
        
        male_acc = male_correct / np.sum(male_mask) if np.sum(male_mask) > 0 else 0
        female_acc = female_correct / np.sum(female_mask) if np.sum(female_mask) > 0 else 0
        
        plt.figure(figsize=(10, 6))
        accuracies = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Custom Score'],
            'Accuracy': [male_acc, female_acc, (male_acc + female_acc) / 2]
        })
        sns.barplot(x='Gender', y='Accuracy', data=accuracies)
        plt.title('Accuracy by Gender')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(accuracies['Accuracy']):
            plt.text(i, v + 0.02, f'{v:.1%}', ha='center')
            
        plt.tight_layout()
        plt.savefig(f"{self.visualizations_path}gender_accuracy.png")
        plt.close()

    def create_interactive_dashboard(self, df, y_test, y_pred, y_prob=None):
        """Create an interactive HTML dashboard with Plotly"""
        # Gender distribution
        fig1 = px.pie(df, names='gender', title='Gender Distribution in Dataset',
                     color='gender', color_discrete_map={'male': 'blue', 'female': 'pink'})
        
        # Shopping time heatmap by gender
        # Group by hour and gender to get counts
        hour_gender = df.groupby(['start_hour', 'gender']).size().reset_index(name='count')
        fig2 = px.density_heatmap(hour_gender, x='start_hour', y='gender', z='count',
                                 title='Shopping Time by Gender', color_continuous_scale='Viridis')
        
        # Category preferences by gender
        cat_cols = [col for col in df.columns if col.startswith('cat_') and col.endswith('_ratio')]
        if cat_cols:
            cat_data = df.groupby('gender')[cat_cols].mean().reset_index()
            cat_data_melted = pd.melt(cat_data, id_vars=['gender'], value_vars=cat_cols, 
                                    var_name='Category', value_name='Ratio')
            cat_data_melted['Category'] = cat_data_melted['Category'].str.replace('cat_', '').str.replace('_ratio', '')
            
            fig3 = px.bar(cat_data_melted, x='Category', y='Ratio', color='gender',
                         barmode='group', title='Product Category Preferences by Gender',
                         color_discrete_map={'male': 'blue', 'female': 'pink'})
        else:
            fig3 = go.Figure()
            fig3.update_layout(title='No category data available')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create annotation text
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append(dict(
                    showarrow=False,
                    text=str(cm[i, j]),
                    xref='x',
                    yref='y',
                    x=j,
                    y=i,
                    font=dict(size=20, color='white' if cm[i, j] > cm.max()/2 else 'black')
                ))
        
        fig4 = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Female', 'Male'],
                y=['Female', 'Male'],
                colorscale='Blues',
                showscale=True
        ))
        
        fig4.update_layout(
            title='Confusion Matrix',
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='Actual'),
            annotations=annotations
        )
        
        # Create dashboard layout
        dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Gender Distribution', 'Shopping Time by Gender', 
                           'Category Preferences', 'Confusion Matrix'),
            specs=[[{'type': 'domain'}, {'type': 'xy'}],
                  [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # Add pie chart
        for trace in fig1.data:
            dashboard.add_trace(trace, row=1, col=1)
        
        # Add heatmap
        for trace in fig2.data:
            dashboard.add_trace(trace, row=1, col=2)
        
        # Add bar chart
        for trace in fig3.data:
            dashboard.add_trace(trace, row=2, col=1)
        
        # Add confusion matrix
        for trace in fig4.data:
            dashboard.add_trace(trace, row=2, col=2)
        
        # Update layout
        dashboard.update_layout(
            title_text="Gender Prediction Model Dashboard",
            height=900,
            width=1200
        )
        
        # Save to HTML
        dashboard.write_html(f"{self.visualizations_path}dashboard.html")
    
    def train(self, data_path, label_path):
        """Train model with extensive feature engineering and balanced classes"""
        # Load data
        data = pd.read_csv(data_path, header=None, 
                          names=['session_id', 'start_time', 'end_time', 'product_ids'])
        labels = pd.read_csv(label_path, header=None, names=['gender'])
        
        # Combine datasets
        df = pd.concat([data, labels], axis=1)
        
        # Create initial data visualizations
        print("Creating data distribution visualizations...")
        self.visualize_data_distribution(df)
        
        # Feature extraction
        df = self.extract_advanced_features(df)
        
        # Select features
        numeric_features = [
            'session_duration', 'start_hour', 'day_of_week', 
            'total_products', 'unique_categories', 'category_diversity'
        ]
        
        # Add category ratio features
        cat_ratio_features = [col for col in df.columns if col.startswith('cat_') and col.endswith('_ratio')]
        numeric_features.extend(cat_ratio_features)
        
        categorical_features = ['part_of_day', 'is_weekend', 'product_cats']
        
        # Prepare feature matrices
        X = df[numeric_features + categorical_features]
        y = df['gender']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Update the categorical transformer definition
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('to_string', FunctionTransformer(lambda x: pd.Series(x.ravel()).astype(str))),
            ('tfidf', TfidfVectorizer(
                preprocessor=lambda x: str(x),
                token_pattern=r'[A-Za-z0-9]+',  # Modified to capture alphanumeric tokens
                min_df=1,  # Changed from 5 to 1 to include all tokens
                stop_words=None  # Don't remove any stop words
            ))
        ])
        
        # Update the preprocessor definition
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat_part_day', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='afternoon')),
                    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                ]), ['part_of_day']),
                ('cat_weekend', Pipeline([
                    ('bool_to_int', FunctionTransformer(lambda x: x.astype(int))),
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
                ]), ['is_weekend']),
                ('cat_products', categorical_transformer, ['product_cats'])
            ],
            remainder='drop'
        )
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Transform the data
        X_train_transformed = preprocessor.fit_transform(X_train)
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
        
        # Define the custom scorer
        custom_scorer = make_scorer(self.custom_score)
        
        # Define and train the model with hyperparameter tuning
        parameters = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        clf = RandomForestClassifier(random_state=42)
        
        # Create a grid search with 3-fold cross-validation
        # Use the custom scorer for evaluation
        grid_search = GridSearchCV(
            clf, parameters, cv=StratifiedKFold(n_splits=3), 
            scoring=custom_scorer, n_jobs=-1, verbose=1
        )
        
        # Fit the grid search
        print("Training model with grid search...")
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Save the preprocessor and best model
        self.preprocessor = preprocessor
        self.model = best_model
        
        # Make predictions
        X_test_transformed = preprocessor.transform(X_test)
        y_pred = best_model.predict(X_test_transformed)
        
        # Get prediction probabilities if available
        try:
            y_prob = best_model.predict_proba(X_test_transformed)
        except:
            y_prob = None
        
        # Calculate scores
        acc_score = accuracy_score(y_test, y_pred)
        custom_score_val = self.custom_score(y_test, y_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy Score: {acc_score}")
        print(f"Custom Score: {custom_score_val}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create model evaluation visualizations
        print("Creating model performance visualizations...")
        self.visualize_model_performance(y_test, y_pred, y_prob)
        
        # Create feature importance visualization
        print("Visualizing feature importance...")
        self.visualize_feature_importance()
        
        # Create interactive dashboard
        print("Creating interactive dashboard...")
        self.create_interactive_dashboard(df, y_test, y_pred, y_prob)
        
        print(f"All visualizations saved to {self.visualizations_path}")
        
        return self.model
    
    def predict(self, data):
        """Make predictions on new data"""
        if self.model is None:
            raise Exception("Model not trained yet. Call train() first.")
        
        # Extract features
        featured_data = self.extract_advanced_features(data)
        
        # Select features
        numeric_features = [
            'session_duration', 'start_hour', 'day_of_week', 
            'total_products', 'unique_categories', 'category_diversity'
        ]
        
        # Add category ratio features
        cat_ratio_features = [col for col in featured_data.columns if col.startswith('cat_') and col.endswith('_ratio')]
        numeric_features.extend(cat_ratio_features)
        
        categorical_features = ['part_of_day', 'is_weekend', 'product_cats']
        
        X = featured_data[numeric_features + categorical_features]
        
        # Transform and predict
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict(X_transformed)

# Main execution
if __name__ == "__main__":
    predictor = GenderPredictor()
    model = predictor.train('/Users/tewoflosgirmay/Desktop/assignments/trainingData.csv', '/Users/tewoflosgirmay/Desktop/assignments/trainingLabels.csv')