
from covar_sgd import Covar
import numpy as np
import pandas as pd
from glob import glob
import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import cnames

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
            covars.append(Covar.load(filename))
    
        return CovarAnalyzer(covars,dataframe)
    
    
    def plotMetric(self,metric,add_legend = False):
        
            
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
                    
                    
        for i,covar in enumerate(self.covars):
            if(add_legend):
                plt.plot(covar.epoch_ind_log,metric(covar),plot_style[i])
            else:
                label = ','.join(f'{column}_{self.dataframe.iloc[i][column]}' for column in self.dataframe.columns if column != 'filename')
                plt.plot(covar.epoch_ind_log,metric(covar),label=label)
        
        #plt.show()
        plt.legend()
        
    def plotCosineSim(self):
        #cosine_sim_metric = lambda covar: np.abs(covar.cosine_sim_log)
        cosine_sim_metric = lambda covar : np.abs([np.mean(np.sqrt(np.sum(covar.cosine_sim_log[i] ** 2,axis = 0))) for i in range(len(covar.cosine_sim_log))])
        self.plotMetric(cosine_sim_metric)
        
    def plotCostval(self):
        cosine_sim_metric = lambda covar: np.log10(covar.cost_log)
        self.plotMetric(cosine_sim_metric)
    
    
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
    
    c = CovarAnalyzer.load('data/results/results.csv')
    #c.plotCosineSim()
    #c.plotCostval()
    
    c.get('learning_rate == 10 and momentum == 0.9').plotCosineSim()
    