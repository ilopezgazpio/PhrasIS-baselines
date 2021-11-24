import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def CorrelationMatrix (df: pd.DataFrame, name : str):

    columns = ['jaccard_strip_tokenized', 'jaccard_strip_tokenized_noPunct_lemmat_noStopWords',
               'jacckard_strip_tokenized_noPunct',
               'left-right', 'right-left', 'path_similarity', 'lch_similarity_nouns', 'lch_similarity_verbs',
               'jcn_similarity_brown_nouns', 'jcn_similarity_brown_verbs', 'jcn_similarity_genesis_nouns',
               'jcn_similarity_genesis_verbs',
               'wup_similarity', 'path_similarity_root', 'lch_similarity_nouns_root', 'lch_similarity_verbs_root',
               'wup_similarity_root',
               'chunk1>chunk2', 'chunk2>chunk1', '|chunk1-chunk2|', 'minimum_difference', 'maximum_difference']

    corr = df[columns].corr()

    sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )

    plt.title(name)
    plt.show()

def CorrelationMatrix2(df: pd.DataFrame, name : str):

    columns = ['jaccard_strip_tokenized', 'jaccard_strip_tokenized_noPunct_lemmat_noStopWords',
               'jacckard_strip_tokenized_noPunct',
               'left-right', 'right-left', 'path_similarity', 'lch_similarity_nouns', 'lch_similarity_verbs',
               'jcn_similarity_brown_nouns', 'jcn_similarity_brown_verbs', 'jcn_similarity_genesis_nouns',
               'jcn_similarity_genesis_verbs',
               'wup_similarity', 'path_similarity_root', 'lch_similarity_nouns_root', 'lch_similarity_verbs_root',
               'wup_similarity_root',
               'chunk1>chunk2', 'chunk2>chunk1', '|chunk1-chunk2|', 'minimum_difference', 'maximum_difference']
    corr = df[columns].corr()

    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs()
    )

    plt.title(name)
    plt.show()


def heatmap(x, y, size):
    fig, ax = plt.subplots()

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = 50
    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

def scatterMatrix(df: pd.DataFrame, name : str):

    col_list = ['jaccard_strip_tokenized', 'jaccard_strip_tokenized_noPunct_lemmat_noStopWords',
                'jacckard_strip_tokenized_noPunct', 'left-right', 'right-left', 'path_similarity',
                'lch_similarity_nouns', 'lch_similarity_verbs', 'jcn_similarity_brown_nouns',
                'jcn_similarity_brown_verbs', 'jcn_similarity_genesis_nouns', 'jcn_similarity_genesis_verbs',
                'wup_similarity', 'path_similarity_root', 'lch_similarity_nouns_root', 'lch_similarity_verbs_root',
                'wup_similarity_root', 'chunk1>chunk2', 'chunk2>chunk1', '|chunk1-chunk2|', 'minimum_difference',
                'maximum_difference']
    # col_list = ["path_similarity", "lch_similarity_nouns", "lch_similarity_verbs"]
    Axes = pd.plotting.scatter_matrix(df [col_list], alpha=0.2, figsize=(10, 10), s=100)

    # y ticklabels
    [plt.setp(item.yaxis.get_majorticklabels(), 'size', 3) for item in Axes.ravel()]
    # x ticklabels
    [plt.setp(item.xaxis.get_majorticklabels(), 'size', 3) for item in Axes.ravel()]
    # y labels
    [plt.setp(item.yaxis.get_label(), 'size', 0.5) for item in Axes.ravel()]
    # x labels
    [plt.setp(item.xaxis.get_label(), 'size', 0.5) for item in Axes.ravel()]

    plt.title(name)
    plt.show()
