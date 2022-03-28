
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec



def stsubplot(fig,position,number,gs=None):
    '''
    create subplot axis
    @params:
    fig: figure object
    position: the position of the subplot axis
    gs: gridSpec
    @return ax

    '''

    axprops = dict(xticks=[], yticks=[])
    width_p=position.x1-position.x0; height_p=(position.y1-position.y0)/number
    left_p=position.x0;bottom_p=position.y1-height_p;
    ax=[]
    for idx in range(number):
        ax.append(fig.add_axes([left_p,bottom_p-idx*height_p,width_p,height_p], **axprops))
        #ax.append(brokenaxes(xlims=((76, 116), (146, 160)), hspace=.05, despine=True,fig=fig,subplot_spec=gs))
        ax[-1].set_xticks([])
        ax[-1].set_xticklabels(labels=[])
    return ax






def gait_diagram(fig,axs,gs,gait_data):
    '''
    plot gait diagram using while and black block to indicate swing and stance phase

    @Params:
    fig: is a figure object of matplotlib
    axs: is a axis object
    gs: gridSpec object
    gait_data is gait phase (M*N), M steps * N legs

    '''
    position=axs.get_position()
    axs.set_yticks([])
    axs.set_yticklabels(labels=[])
    axs.set_xticks([])
    axs.set_xticklabels(labels=[])
    #axs.set_title("Gait",loc="left",pad=2)



    # colors
    c4_1color=(46/255.0, 77/255.0, 129/255.0)
    c4_2color=(0/255.0, 198/255.0, 156/255.0)
    c4_3color=(255/255.0, 1/255.0, 118/255.0)
    c4_4color=(225/255.0, 213/255.0, 98/255.0)
    colors=[c4_1color, c4_2color, c4_3color, c4_4color]
    cmap = (mpl.colors.ListedColormap(['white', 'cyan', 'yellow', 'royalblue']).with_extremes(over='red', under='blue'))
    ax=stsubplot(fig,position,4,gs)
    xx=[]
    LegName=['RF','RH', 'LF', 'LH']
    #barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest',vmin=0.0,vmax=1.0)
    barprops = dict(aspect='auto', cmap=cmap, interpolation='nearest',vmin=0.0,vmax=1.0)
    for idx in range(4):
        ax[idx].set_yticks([0.1*(idx+1)])
        xx.append(np.where(gait_data[:,idx]>0.2*max(gait_data[:,idx]),1.0,0.0)) # > 0.2 times of max_GRF, then leg on stance phase
        ax[idx].imshow(xx[idx].reshape((1,-1)),**barprops)
        ax[idx].set_ylabel(LegName[idx])
        ax[idx].set_yticklabels(labels=[])





def plot_gait_diagram():
    '''
    example to plot gait diagram

    '''
    figsize=(6,6.5+2)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs=gridspec.GridSpec(2,1)#13
    axs=[]
    axs.append(fig.add_subplot(gs[0:2,0]))

    
    # for example data
    gait_data=np.array([
        [1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0],
        [1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0]]
    )
    gait_data=gait_data.T

    gait_diagram(fig,axs[0],gs,gait_data)
    plt.show()


if __name__=="__main__":

    plot_gait_diagram()

