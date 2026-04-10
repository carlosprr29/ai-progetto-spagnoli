import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

def plot_class_balance(df, target_col='label'):
    print("\n Generating class balance chart...")
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x='label', palette='viridis')
    plt.title('News Distribution (0: Real, 1: Fake)')
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.show()

def generate_cloud(df, news_label, text_col='title', colour_map='viridis', graph_title='Word Cloud'):
    """
    Generates word clouds based on the news label.
    Args:
        df: DataFrame
        news_label: 0 (Real) or 1 (Fake)
        text_col: Column to analyze (usually 'title' or 'text')
        colour_map: Matplotlib colormap (e.g., 'Greens', 'Reds')
        graph_title: Title for the plot
    """
    # Filtering the text based on the label (0 or 1)
    # We use 'label' here because that's the column name in your WELFake dataset
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
    plt.title(graph_title, fontsize=16)
    plt.axis('off')
    plt.show()

def analyse_term(df, term, col='title'):
    filtered_df = df[df[col].str.contains(term, case=False, na=False)]
    
    if not filtered_df.empty:
        count = filtered_df['label'].value_counts(normalize=True) * 100
        
        dist_dict = count.to_dict()
        
        print(f"The term '{term}' appears in {len(filtered_df)} {col}s. Distribution: {dist_dict}")
    else:
        print(f"The term '{term}' was not found in {col}.")
        
        

def plot_text_length(df, column='text', bins=100, max_chars=5000):
    """
    Plots a histogram comparing the length of texts between Real and Fake news.
    
    Args:
        df (pd.DataFrame): The dataset containing the news.
        column (str): The column to analyze (default is 'text').
        bins (int): Number of bins for the histogram.
        max_chars (int): The x-axis limit (texts are much longer than titles).
    """
    print(f"\n Analysing {column} length...")
    
    # Creamos la columna temporal de longitud
    temp_df = df.copy()
    df['temp_len'] = df[column].apply(len)

    temp_df['label_name'] = temp_df['label'].map({0: 'Real', 1: 'Fake'})    

    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df[df['temp_len'] <= max_chars], # Filtramos aquí para que el KDE sea preciso
        x='temp_len', 
        hue='label', 
        kde=True, 
        bins=bins, 
        palette='magma'
    )
    
    plt.title(f'Comparison of {column.capitalize()} Length: Real vs Fake')
    plt.xlabel('Number of characters')
    plt.ylabel('Frequency')
    plt.xlim(0, max_chars) 
     
    plt.show()
    
    # Borramos la columna temporal
    df.drop(columns=['temp_len'], inplace=True)
    
def plot_ablation_results(trainer_tit, trainer_full):
    """
    Generates a comparison plot between Titles Only and Full Content models
    using the evaluation accuracy stored in the Trainer log history.
    """

    print("Generating final ablation comparison...")

    # Extract accuracy from trainer logs
    res_tit = trainer_tit.state.log_history[-2].get('eval_accuracy')
    res_full = trainer_full.state.log_history[-2].get('eval_accuracy')


    # Create visualization
    plt.figure(figsize=(9, 6))
    colors = ['#A0C4FF', '#023E8A']
    bars = plt.bar(['Titles Only', 'Full Content'], [res_tit, res_full], color=colors)

    # Styling
    plt.ylim(0.85, 1.0)
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

    print(f"Comparison complete: Full context improves accuracy by {(res_full - res_tit)*100:.2f}%")


    
    