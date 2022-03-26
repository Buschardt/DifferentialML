import numpy as np
import matplotlib.pyplot as plt

def plotTests(S0, S0_test, y, y_test, label, y_true=None,
 figsize=None, cols=None, rows=None, error=False, model='Black-Scholes'):
    
    #input:
        #- X indeholder dine S0'er
        #- X_test indeholder S0_test
        #- y indeholder simulerede C og greeks
        #- y_test indeholder approksimerede y og greeks, samt sande værdier?
        #- ellers sande værdier for sig
            #- sande værdier for sig er en god idé da vi ikke har dem med i LSM plots
            #- lav en if y_true != None så plot sande værdier

    nPlots = y_test.shape[1]
    #hvis cols/rows er none så lav selv grid
    if cols == None:
        if nPlots <= 3:
            cols = nPlots
            rows = 1
        else:
            cols = 2
            if (nPlots % 2) == 0:
                rows = nPlots / 2
            else:
                rows = round(nPlots / 2) + 1

    #figsize hvis none, lav selv størrelse
    if figsize == None:
        figsize = [4*cols+1.5, 4*rows]
    

    #dobbelt for loop der danner plots
    #for i in cols
    #for j in rows
    plt.figure(figsize=figsize)
    for i in range(0, nPlots):
        plt.subplot(rows, cols, i+1)
        plt.plot(S0, y[:,i], 'o', color='grey', label=f'Simulated {label[i]}', alpha = 0.3)
        plt.plot(S0_test, y_test[:,i], color='red', label='NN approximation')
        if type(y_true) is np.ndarray:
            plt.plot(S0_test, y_true[:,i], color='black', label=f'{model} {label[i]}')
        plt.xlabel('S0')
        plt.ylabel(f'{label[i]}')
        plt.legend()
        if label[i] == 'vega':
            y_min = - 1.5
            y_max = y_test[:,i].max() + 1
            plt.ylim([y_min, y_max])
        plt.title(f'Differential ML - {label[i]} approximation')

    plt.tight_layout()
    plt.show()

    #osv
    #variabel fro save plot?

    #husk error plot
    if error == True:
        plt.figure(figsize=figsize)
        for i in range(0, nPlots):
            error_i = y_true[:,i] - y_test[:,i]
            RMSE_i = np.sqrt((error_i**2).mean())
            RMSE_i_format = "{:.6f}".format(RMSE_i)
            plt.subplot(rows, cols, i+1)
            plt.plot(S0_test, error_i, color='red', label = 'Predicted')
            plt.plot(S0_test, [0]*y_true.shape[0], color = 'black', label='Actual')
            plt.plot([], [], ' ', label=f'RMSE: {RMSE_i_format}')
            plt.xlabel('S0')
            plt.legend()
            plt.title(f'{label[i]} - Error')
        
        plt.tight_layout()
        plt.show()