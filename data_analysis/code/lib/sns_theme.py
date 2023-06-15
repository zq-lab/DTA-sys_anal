import seaborn as sns
#字体和坐标轴的线，等的刻度大小，唯一的参数是字号大小，根据自己的需要增加参数，下面这些都是TJxb的线宽标准（也可能有错）
def sns_context(fontsize):
    conText={
        'axes.linewidth'   : 0.75, #坐标轴线宽为0.75
        'grid.linewidth'   : 0.75, #网格线宽为0.75
        'lines.linewidth'  : 1.0, #绘图线宽1.0
        'lines.markersize' : 3.0, #散点的大小3.0
        'patch.linewidth'  : 1.0, #路径线宽1.0
        'xtick.major.width': 0.75, #下面这是主副刻度线的宽度
        'ytick.major.width': 0.75,
        'xtick.minor.width': 0.75,
        'ytick.minor.width': 0.75,
        'xtick.major.size' : 2,
        'ytick.major.size' : 2,
        'xtick.minor.size' : 1.2,
        'ytick.minor.size' : 1.2,
        'font.size'        : 7.5,#字号
        'axes.labelsize'   : fontsize,#xy坐标轴的字号
        'axes.titlesize'   : fontsize,
        'xtick.labelsize'  : fontsize,
        'ytick.labelsize'  : fontsize,
        'legend.fontsize'  : fontsize,
        'legend.title_fontsize': fontsize
    }
    return conText
