import os
import sys
import csv 
import pickle
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
from Up_Lower_Bound import UpperLowerBound
from tensorflow.keras import callbacks
tf.random.set_seed(2)
# np.random.seed(2)

class Run:

   epochs = 2000
   batch_size = 256
   #Stop training when a monitored metric has stopped improving
   early_stopping = callbacks.EarlyStopping(
      monitor = 'coverage_width_rate',
      min_delta=0.0001,  # an absolute change of less than min_delta, will count as no improvement.
      patience=30,      # 0.1* epoches Number of epochs with no improvement after which training will be stopped
      restore_best_weights=True
   )

   def __init__(self, No_PCGrad, PCGrad):
      self.No_PCGrad = No_PCGrad
      self.PCGrad = PCGrad
      self.No_PCGrad_model = No_PCGrad.model 
      self.PCGrad_model = PCGrad.model
      # To keep the results 
      self.result = []
      self.opt = optimizers.Adam()

   @classmethod
   def set_epochs(cls, epoches):
      cls.epochs = epoches

   @classmethod
   def set_batch_size(cls, batch_size):
      cls.batch_size = batch_size

   def run_no_pcgrad(self):

      self.No_PCGrad_model.init_arguments()
      self.No_PCGrad_model.compile(optimizer=self.opt,
      loss = [self.No_PCGrad_model.selective_up, self.No_PCGrad_model.selective_low, self.No_PCGrad_model.up_penalty, self.No_PCGrad_model.low_penalty, self.No_PCGrad_model.coverage_penalty],
      metrics = [self.No_PCGrad_model.coverage, self.No_PCGrad_model.mpiw, self.No_PCGrad_model.coverage_width_rate])
      history_no_pcgrad = self.No_PCGrad_model.fit(self.No_PCGrad.X_train, self.No_PCGrad.y_train, 
      validation_data = (self.No_PCGrad.X_test, [self.No_PCGrad.y_test[:,0], self.No_PCGrad.y_test[:,1], self.No_PCGrad.y_test[:,-1]]),
      batch_size=self.batch_size, 
      epochs= self.epochs,
      #callbacks=[Run.early_stopping], 
      verbose=0)

      # Save the training history 
      name = self.No_PCGrad.dataset.split('.')[0]
      with open(f'{name}_history.pkl', 'wb') as handle:
         pickle.dump(history_no_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
      # To plot and save .png
      self.plot_training(f'{name}_history.pkl')

      no_pcgrad_pred = self.No_PCGrad_model.predict(self.No_PCGrad.X_test)
      df_raw = pd.DataFrame(no_pcgrad_pred, columns = ['Lowerbound', 'Upperbound', 'G_X'])

      # Transfored to original y
      no_pcgrad_pred = self.No_PCGrad.scaler_y.inverse_transform(no_pcgrad_pred)
      df = pd.DataFrame(no_pcgrad_pred, columns = ['Lowerbound', 'Upperbound', 'G_X'])
      #### attention!!
      df['G_X'] = df_raw['G_X']
      df['y_true'] = self.No_PCGrad.reversed_data(self.No_PCGrad.y_test[:,0].reshape(-1,1))
      df['Width'] = (df['Upperbound']-df['Lowerbound'])
      df['MPIW'] = np.mean(df['Width'])
      df['NMPIW'] = df['MPIW']/self.No_PCGrad.range
      df['Flag']= np.where((df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0)  
      df['PICP'] =  np.mean( (df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']) )

      # To sort the columns 
      df = df[['PICP','Lowerbound','y_true','Upperbound','Flag','MPIW','NMPIW','Width','G_X']]
      
      # Save all predicted value
      with open(f'{name}_pred.csv', 'w') as f:
         df.to_csv(f, header=True, float_format = float, index = False)
  
      self.result.append({'PICP':np.mean(df['PICP']),'MPIW':np.mean(df['MPIW']),'NMPIW':np.mean(df['NMPIW'])})

   def run_pcgrad(self):
      self.PCGrad_model.init_arguments(method = 'PCGrad')

      self.PCGrad_model.compile(optimizer=self.opt,
      loss = [self.PCGrad_model.selective_up, self.PCGrad_model.selective_low, self.PCGrad_model.up_penalty, self.PCGrad_model.low_penalty, self.PCGrad_model.coverage_penalty],
      metrics = [self.PCGrad_model.coverage, self.PCGrad_model.mpiw, self.PCGrad_model.coverage_width_rate])

      history_pcgrad = self.PCGrad_model.fit(self.PCGrad.X_train, self.PCGrad.y_train, 
      validation_data = (self.PCGrad.X_test, [self.PCGrad.y_test[:,0], self.PCGrad.y_test[:,1], self.PCGrad.y_test[:,-1]]),
      batch_size=self.batch_size, 
      epochs= self.epochs,
      #callbacks=[Run.early_stopping], 
      verbose=0)

      # Save the training history 
      name = self.PCGrad.dataset.split('.')[0]
      with open(f'{name}_pcgrad_history.pkl', 'wb') as handle:
         pickle.dump(history_pcgrad.history, handle, protocol=pickle.HIGHEST_PROTOCOL)     
      #model.save_weights("checkpoints/{}".format(self.filename))

      self.plot_training(f'{name}_pcgrad_history.pkl')

      pcgrad_pred = self.PCGrad_model.predict(self.PCGrad.X_test)
      df_raw = pd.DataFrame(pcgrad_pred, columns = ['Lowerbound', 'Upperbound', 'G_X'])

      # Transfored to original y
      pcgrad_pred = self.PCGrad.scaler_y.inverse_transform(pcgrad_pred)
      df = pd.DataFrame(pcgrad_pred, columns = ['Lowerbound', 'Upperbound', 'G_X'])

      df['G_X'] = df_raw['G_X']
      df['y_true'] = self.PCGrad.reversed_data(self.PCGrad.y_test[:,0].reshape(-1,1))
      df['Width'] = (df['Upperbound']-df['Lowerbound'])
      df['MPIW'] = np.mean(df['Width'])
      df['NMPIW'] = df['MPIW'] / self.PCGrad.range
      df['Flag']= np.where((df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']), 1, 0) 
      df['PICP'] = np.mean( (df['Upperbound'] >= df['y_true']) & (df['Lowerbound'] <= df['y_true']))
      
      df = df[['PICP','Lowerbound','y_true','Upperbound','Flag','MPIW', 'NMPIW', 'Width','G_X']]

      # Save all predicted value 
      with open(f'{name}_pcgrad_pred.csv', 'w') as f:
         df.to_csv(f, header=True, float_format = float, index = False)
      
      self.result.append({'PICP':np.mean(df['PICP']), 'MPIW':np.mean(df['MPIW']), 'NMPIW':np.mean(df['NMPIW'])})

   def print_comparison(self):
      res = pd.DataFrame(self.result, index = ['No_PCGrad', 'PCGrad'])
      print(res)
      return res

   def plot_training(self, filename):
      dict_data = pd.read_pickle(filename)  
      df = pd.DataFrame(dict_data)
      title = f'{filename}'
      fig = plt.figure(figsize=(10,6))
      sns.set_style("ticks")
      plt.title(title)
      plt.xlabel("Epochs")
      sns.lineplot(data=df[ ['coverage', 'mpiw','val_coverage', 'val_mpiw','val_loss']])
      plt.savefig(f'{title}.png', dpi = 600)
      plt.clf()
      #plt.show()
     

def main():

   # # # for datasets1
   # datasets1 = ['1_constant_noise.csv','2_nonconstant_noise.csv','4_Concrete_Data.xls','5_BETAPLASMA.csv', '6_Drybulbtemperature.xlsx', '7_moisture content of raw material.xlsx', 
   # '8_steam pressure.xlsx', '9_main stem temperature.xlsx','10_reheat steam temperature.xlsx']
   # targets1 = ['y', 'y', 'Concrete compressive strength(MPa, megapascals) ','BETAPLASMA','dry bulb temperature',
   # 'moisture content of raw material','steam pressure','main stem temperature','reheat steam temperature']

   datasets1 = ['4_Concrete_Data.xls']
   targets1 = ['Concrete compressive strength(MPa, megapascals) ']

   # # for datasets2
   # datasets2  = ['1_Boston_Housing.csv', '2_Concrete_Data.xls',
   # '3_Energy Efficiency.csv', '4_kin8nm.csv', '5_Naval Propulsion.csv', 
   # '6_Power.csv', '7_Protein.csv', '8_Wine Quality.csv', '9_Yacht.csv','10_Song_Year.csv']

   # targets2 = ['MEDV','Concrete compressive strength(MPa, megapascals) ','Y1','y',
   # 'gt_t_decay','Net hourly electrical energy output','y','quality','Residuary resistance per unit weight of displacement',
   # 'Year']


   def training_data(datasets, targets):
      for index, (dataset, target) in enumerate(zip(datasets, targets)):
         if dataset == '7_Protein.csv':
            times = 5 
         elif dataset == '10_Song_Year.csv':
            times = 1
         else:
            times = 1
         temp = []
         for i in range(times):
            No_PCGrad = UpperLowerBound(dataset, target)
            PCGrad = UpperLowerBound(dataset, target)
            obj = Run(No_PCGrad, PCGrad)
            obj.run_no_pcgrad()
            obj.run_pcgrad()
            res = obj.print_comparison()
            temp.append(res)
         output = pd.concat(temp)
         output.to_csv(f'output_{index+1}.csv')

   training_data(datasets1, targets1)
   #training_data(datasets2, targets2)

if __name__ == "__main__":
   # set the epochs 
   Run.epochs = 2000
   # set the batch_size
   # Run.batch_size = 1000
   main()















