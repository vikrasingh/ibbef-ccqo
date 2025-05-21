import numpy as np
import matplotlib.pyplot as plt

def main(data, linespecs=None, linewidth=1.6,legendnames=None,
         fontsize=18, tickfontsize=14, legendfontsize=14):
    """ create boxplots
    
    """
    box_colors=['red','blue','green']
    n=np.shape(data)
    positions = np.arange(1, n[0] + 1)  # Define positions for the boxes
    plt.figure()
    for i in range(n[1]):
        plt.boxplot(data[:,i],positions=[positions[i]],patch_artist=True,
                 boxprops={'facecolor': box_colors[i], 'edgecolor': 'black', 'linewidth': 1.5},
                 whiskerprops={'color': 'black', 'linewidth': 1,'linestyle':'--'},
                 capprops={'color': 'black', 'linewidth': 1.5},
                 medianprops={'color': 'black', 'linewidth': 2},
                 flierprops={'marker': '+', 'markersize': 6, 'markerfacecolor': 'black'},
                 showmeans=False,
                 meanprops={'marker': 'D', 'markeredgecolor': 'black', 'markerfacecolor': 'lightgreen'})
    
    plt.ylabel('RSS')
    plt.title(f'BSS')
    plt.grid(True)
    plt.xticks([1, 2, 3], ['IBB','IBBEF', 'MIO'])
    plt.draw() # show the plot   
    
    # save the plot as png
    fig_handle=plt.gcf()   # get the handle to the current fig
    #fig_handle.savefig("bp_relgap.png")

    # save to PDF
    fig_handle.savefig("bp_relgap.pdf")
    