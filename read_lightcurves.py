import pickle

#Right now, this script only takes light curve data and graphs it
#But later, we can incorporate the completeness thing 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 




mpl.style.use('ggplot')
mpl.rc('axes', prop_cycle=mpl.cycler(color=["#55A868", "#C44E52",
                            "#8172B2", "#CCB974"]))
color_dict = {'u': "#55A868", 'g':"#C44E52", 'i':"#8172B2", 'z':"#CCB974"}

# What range of lightcurves do you want to read in values for 
for idindex in range(0, 1):
   # This is the standardized phi range ig 
   for phi in np.linspace(0, 0.9, 10):
      input_dict = {}
      fig, ax = plt.subplots()
      filename = f"generated_curves/lc{idindex}phi{phi:.1f}.p"
      input_dict = pickle.load( open(filename, "rb" ))

      #Graph the values to confirm that nothing is screwed up when reading in values 
      for band in 'ugiz':
         #ax.scatter(phase[bands==band], mags[bands==band], marker='.', color=color_dict[band])
         #ax.errorbar(real_phases, real_mags, error, fmt='s', label=band, color=color_dict[band])
         
         ax.scatter(input_dict[band]["PHASES"], input_dict[band]["MAGS"],marker='o', color=color_dict[band], label=band)
         
      

      
      
      ax.set(xlabel='time (phase))', ylabel='mag',
         title='lcid={0} generated lightcurves'.format(idindex))
      ax.legend(loc='upper left', ncol=5, numpoints=1)
      plt.show()