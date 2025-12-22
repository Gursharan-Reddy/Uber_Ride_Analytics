import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(importance, save_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importance.values, 
        y=importance.index, 
        hue=importance.index, 
        palette='viridis', 
        legend=False
    )
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_actual_vs_predicted(daily_df, test_indices, y_pred, save_path):
    results = pd.DataFrame({
        'Date': daily_df.loc[test_indices, 'date_only'],
        'Actual': daily_df.loc[test_indices, 'trip_count'],
        'Predicted': y_pred
    }).sort_values('Date')

    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], results['Actual'], label='Actual', marker='o', alpha=0.7)
    plt.plot(results['Date'], results['Predicted'], label='Predicted', linestyle='--', marker='x', alpha=0.7)
    plt.title('Actual vs Predicted Daily Trips')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()