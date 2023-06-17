# ploting functions

from matplotlib import cm
import matplotlib.pyplot as plt
import joypy
import seaborn as sns



# plot heatmap based on matched x and y
def heatmap_value_counts(df, x, y, count_thd=100, fillna=0, blank_str_replace='Other', cmap='Blues', figsize=(10,7)):
    sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.0)
    data = df.value_counts(subset=[x,y]).reset_index()
    data.replace(to_replace=r'^\s*$',value=blank_str_replace,regex=True,inplace=True)
    data.sort_values(by='count',ascending=False)
    data = data[data['count']>=count_thd].pivot(index=y, columns=x,values='count')
    data.fillna(0,inplace=True)
    data = data.astype(int)
    # data.sort_values(by='count', ascending=False)

    # f, ax = plt.subplots(figsize=(10, 7))
    f, ax = plt.subplots(figsize=figsize)
    return sns.heatmap(data, annot=True, fmt="d", linewidths=.5, ax=ax, cmap=cmap, alpha=0.9)

# joy plot optimized
def joy(data, x, y, x_range=None, blank_str_replace='Other'):
    sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.2)
    fig, axes = joypy.joyplot(data, column=x, by=y,
                                figsize=(15,6),
                                ylim = 'own',
                                x_range = x_range,
                                colormap = cm.Paired,
                                overlap = 0.1,
                                alpha = 0.5)
    return fig

# bar with values
def bar(data, x, y, xlim=None, isLable=True):
    sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.2)
    fig = sns.barplot(data=data, x=x, y=y, alpha=0.8)
    if xlim is not None:
        fig.set_xlim(xlim)
    if isLable:
        for i in fig.containers:
            fig.bar_label(i,)
    return fig

# relplot based on counts of matched x and y
def rel_count(df, x, y, count_thd = 1, title=None, height=5,figsize=None):
    sns.set_theme(style="whitegrid")
    data = df.value_counts(subset=[x,y]).reset_index()
    data.fillna(0,inplace=True)
    # data = data.astype(int)
    data.sort_values(by=y)

    g = sns.relplot(
            data=data[data['count'] >= count_thd],
            x=x, y=y, hue="count", size="count",
            palette="vlag",      
            hue_norm=(count_thd, data['count'].max()),
            edgecolor="1",
            height=8
        )

    if figsize is not None:
        g.figure.set_size_inches(figsize)

    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    g.set_xticklabels(rotation=90)
    if title is not None:
        g.set(title=title)

    return g


# only prepared data frame
def stripe(df, x, y, title = '', height=5, figsize=None):
    sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.2)
    g = sns.PairGrid(df, x_vars=x, y_vars=y, height=height)
    g.map(sns.stripplot, size=8, orient="h", jitter=False, palette="Blues", linewidth=1, edgecolor="w")

    for ax in g.axes.flat:
        ax.set(title=title)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    if figsize is not None:
        g.figure.set_size_inches(figsize)
    
    return g