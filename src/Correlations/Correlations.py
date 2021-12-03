import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def CorrelationMatrix (df: pd.DataFrame, columns : list, name : str, fillNA=False, savePath=None):
    ''' Color map correlation matrix '''

    corr = df[columns].corr()

    if fillNA:
        corr = corr.fillna(0)

    sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )

    plt.title(name)

    if savePath:
        plt.savefig(savePath,dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def CorrelationMatrix2(df: pd.DataFrame, columns : list, name : str, fillNA=False, savePath=None):
    ''' Square size correlation matrix '''

    corr = df[columns].corr()

    if fillNA:
        corr = corr.fillna(0)

    corr = pd.melt(corr.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']

    heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs()
    )

    plt.title(name)

    if savePath:
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def heatmap(x, y, size):
    fig, ax = plt.subplots()

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = 50
    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * size_scale,
        marker='s'
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

def scatterMatrix(df: pd.DataFrame, columns : list, name : str, fillNA=False, savePath=None):
    ''' Scatter plot '''

    Axes = pd.plotting.scatter_matrix(df [columns], alpha=0.2, figsize=(10, 10), s=100)

    # y ticklabels
    [plt.setp(item.yaxis.get_majorticklabels(), 'size', 4) for item in Axes.ravel()]
    # x ticklabels
    [plt.setp(item.xaxis.get_majorticklabels(), 'size', 4) for item in Axes.ravel()]
    # y labels
    [plt.setp(item.yaxis.get_label(), 'size', 6) for item in Axes.ravel()]
    # x labels
    [plt.setp(item.xaxis.get_label(), 'size', 6) for item in Axes.ravel()]

    plt.suptitle(name)

    if savePath:
        plt.savefig(savePath, dpi=300)
    else:
        plt.show()
    plt.close()
