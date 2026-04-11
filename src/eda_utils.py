import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

def plot_class_balance(df):
    """Generates a bar chart to visualize the distribution of Real vs Fake news."""
    print("\n--- Generating class balance chart ---")
    plt.figure(figsize=(7, 5))
    # Standardizing labels for the plot
    plot_df = df.copy()
    plot_df['Label'] = plot_df['label'].map({0: 'Real', 1: 'Fake'})
    
    sns.countplot(data=plot_df, x='Label', palette='viridis')
    plt.title('Dataset Distribution (0: Real, 1: Fake)', fontsize=14)
    plt.ylabel('Count')
    plt.xlabel('Category')
    plt.show()

def generate_cloud(df, news_label, text_col='title', colour_map='viridis', graph_title='Word Cloud'):
    """
    Generates word clouds based on the news label (Real or Fake).
    """
    print(f"\n--- Generating Word Cloud for {'Fake' if news_label==1 else 'Real'} news ---")
    
    # Filtering the text and joining all rows into a single string
    text = " ".join(df[df['label'] == news_label][text_col].astype(str))
    
    # Initialize WordCloud
    wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100, 
        colormap=colour_map
    ).generate(text)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(graph_title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.show()

def analyse_term(df, term, col='title'):
    """Analyzes how many times a specific term appears and its distribution across labels."""
    filtered_df = df[df[col].str.contains(term, case=False, na=False)]
    
    if not filtered_df.empty:
        count = filtered_df['label'].value_counts(normalize=True) * 100
        dist_dict = count.to_dict()
        # Clean dictionary mapping for display
        friendly_dist = {('Fake' if k==1 else 'Real'): f"{v:.2f}%" for k, v in dist_dict.items()}
        
        print(f"Term '{term}' found in {len(filtered_df)} {col}s.")
        print(f"   ↳ Distribution: {friendly_dist}")
    else:
        print(f"Term '{term}' was not found in {col}.")

def plot_text_length(df, column='text', bins=100, max_chars=5000):
    """
    Plots a histogram comparing text length between Real and Fake news.
    """
    print(f"\n--- Analyzing {column} length distribution ---")
    
    # Use a temporary DataFrame to avoid modifying the original one
    temp_df = df.copy()
    temp_df['temp_len'] = temp_df[column].apply(len)
    temp_df['Category'] = temp_df['label'].map({0: 'Real', 1: 'Fake'})

    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=temp_df[temp_df['temp_len'] <= max_chars],
        x='temp_len', 
        hue='Category', # Use the mapped names for the legend
        kde=True, 
        bins=bins, 
        palette='magma'
    )
    
    plt.title(f'Text Length Comparison: Real vs Fake ({column.capitalize()})', fontsize=14)
    plt.xlabel('Number of characters')
    plt.ylabel('Frequency')
    plt.xlim(0, max_chars) 
    plt.show()
    
def plot_ablation_results(trainer_tit, trainer_full):
    """
    Compares the impact of semantic context (Titles vs Full Content) 
    using BERT's evaluation accuracy.
    """
    print("\n--- Generating final ablation study comparison ---")

    # Extract accuracy from trainer logs. 
    # Usually -1 is the summary, -2 is the last evaluation before finish
    try:
        res_tit = next(item['eval_accuracy'] for item in reversed(trainer_tit.state.log_history) if 'eval_accuracy' in item)
        res_full = next(item['eval_accuracy'] for item in reversed(trainer_full.state.log_history) if 'eval_accuracy' in item)
    except Exception as e:
        print(f"⚠️ Error extracting metrics: {e}. Check if training finished correctly.")
        return

    # Create visualization
    plt.figure(figsize=(9, 6))
    colors = ['#A0C4FF', '#023E8A']
    bars = plt.bar(['Titles Only', 'Full Content'], [res_tit, res_full], color=colors)

    # Styling
    plt.ylim(0.80, 1.0) # Starting from 80% to highlight the difference
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('BERT Ablation Study: Impact of Semantic Context on Detection', fontsize=14, fontweight='bold')

    # Add labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005,
                 f"{yval*100:.2f}%",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    improvement = (res_full - res_tit) * 100
    print(f"Ablation complete: Full context yields a {improvement:.2f}% accuracy gain.")