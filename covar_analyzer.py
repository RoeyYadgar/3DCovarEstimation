
from covar_sgd import Covar
from covar_estimation import CovarCost
import numpy as np
import pandas as pd
import torch
from glob import glob
import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import cnames
import mplcursors

from aspire.image import Image


class CovarAnalyzer():
    
    def __init__(self,covars,dataframe = None):
        #super(CovarAnalyzer,self).__init__(dataframe)
        self.covars = covars
        self.dataframe = dataframe
        
    def __getitem__(self,index):
        if(type(index) == int):
            return self.covars[index]
        elif(type(index) == list):
            covars = [self.covars[ind] for ind in index]    
            dataframe = self.dataframe.loc[index]
            return CovarAnalyzer(covars,dataframe)
        else:
            return CovarAnalyzer(self.covars[index],self.dataframe.loc[index][:-1])
        
    def get(self,condition):
        dataframe_bool = self.dataframe.eval(condition)
        index = dataframe_bool.index[dataframe_bool].tolist()
        
        return self.__getitem__(index)
        
    @staticmethod
    def load(input_string):
        #input_string can be either a csv filename or a file pattern for the desired result files
        if(os.path.isfile(input_string)):
            dataframe = pd.read_csv(input_string,index_col = 0)
            filenames = dataframe['filename']
        else:
            filenames = glob(input_string)
            dataframe = None
            
            
        covars = []
        for filename in filenames:
            covars.append(torch.load(filename))
    
        return CovarAnalyzer(covars,dataframe)
    
    
    def plotMetric(self,metric,add_legend = False,xlabel = None,ylabel = None, title = None):
        
            
        #parameter_mapping = [('color'),('linestyle' , ['-','--','-.',':']),('color' , ['red','green','blue'])]
        parameter_mapping = [['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'],list(Line2D.markers),list(Line2D.lineStyles)]
        plot_style = ['' for i in range(len(self.covars))]
        parameter_counter = 0
        if(add_legend):
            for col in self.dataframe.columns:
                if(col != 'filename' and self.dataframe[col].nunique() > 1):
                    unique_ind = pd.factorize(self.dataframe[col])[0]
                    dataframe_style = [parameter_mapping[parameter_counter][uq] for uq in unique_ind]
                    plot_style = [plot_style[i] + dataframe_style[i] for i in range(len(self.covars))]
                    
                    parameter_counter+=1
                    
        ax = plt.axes()    
        
        for i,covar in enumerate(self.covars):
            if(add_legend):
                ax.plot(covar["log_epoch_ind"],metric(covar),plot_style[i])
            else:
                label = ','.join(f'{column}_{self.dataframe.iloc[i][column]}' for column in self.dataframe.columns if (column != 'filename' and len(pd.unique(self.dataframe[column])) > 1))
                ax.plot(covar["log_epoch_ind"],metric(covar),label=label)
                mplcursors.cursor().connect(
                        "add",
                        lambda sel: sel.annotation.set_text(sel.artist.get_label()))
        
        #plt.show()
        ax.legend()
        
        

        if(xlabel != None):
            ax.set_xlabel(xlabel)
        if(ylabel != None):
            ax.set_ylabel(ylabel)
        if(title != None):
            ax.set_title(title)

        return ax
        
    def plotCosineSim(self, title = None):
        cosine_sim_metric = lambda covar : [np.mean(np.sqrt(np.sum(covar['log_cosine_sim'][i] ** 2,axis = 0))) for i in range(len(covar['log_cosine_sim']))]
        return self.plotMetric(cosine_sim_metric,xlabel = 'Epochs',ylabel = 'Mean Cosine Simlarity', title = title)

    def plotWeightedCosineSim(self, title = None):
        singular_vals = lambda covar : np.linalg.norm(covar['vectorsGD'].cpu().numpy().reshape((covar['vectorsGD'].shape[0],-1)),axis=1)
        cosine_sim_metric = lambda covar : [np.sum(np.sqrt(np.sum(covar['log_cosine_sim'][i] ** 2,axis = 0)) * singular_vals(covar))/np.sum(singular_vals(covar)) for i in range(len(covar['log_cosine_sim']))]
        return self.plotMetric(cosine_sim_metric,xlabel = 'Epochs',ylabel = 'Weighted Cosine Simlarity', title = title)
    
    def plotFroErr(self,title = None):
        return self.plotMetric(lambda covar : covar['log_fro_err'],xlabel = 'Epochs',ylabel = 'Frobenium norm error',title = title)
    '''
    def plotCostval(self, title = None):
        cosine_sim_metric = lambda covar: np.log10(covar.cost_log)
        return self.plotMetric(cosine_sim_metric,xlabel = 'Epochs',ylabel = 'Cost Value', title = title)
     
    
    def innprod(self):
        innprod_mat = []
        for covar in self.covars:
            vectors = covar.vectors.detach().numpy().reshape((covar.rank,-1))
            vectors = vectors / np.linalg.norm(vectors,axis=1).reshape((covar.rank,-1))
            innprod_mat.append(np.matmul(vectors,vectors.transpose()))
    
        return innprod_mat
    
    def compareCostToGD(self,reg = 0):
        cost_estimted_vec = []
        cost_ground_truth = []
        for cov in self.covars: 
            images = Image(cov.src.images[:])
            cost_estimted_vec.append(cov.cost(0,images,reg).item())

            cost_ground_truth.append(CovarCost.apply(torch.tensor(cov.vectorsGD.asnumpy()),cov.src,0,images,reg).item())

        return cost_estimted_vec,cost_ground_truth
    '''  
    def updateResultFilesName(self,pattern):
        
        
        def format_string(row):
            return pattern.format(**row)
        
        filenames = self.dataframe.apply(format_string,axis=1)
        
        
        for i in range(len(filenames)):
            current_filename = self.dataframe['filename'][i]
            new_filename = filenames[i]
            print(f'Renaming {current_filename} to {new_filename}')
            os.rename(current_filename,new_filename)
            
    


if __name__ == "__main__":
    
    c = CovarAnalyzer.load('data/rank4_L64_test/results.csv')

    c.plotCosineSim()
    