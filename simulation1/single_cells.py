#Tested with a 2.7 python version#
##################################
# This script shows the response of 
# the single cell models (RS & FS) to a 
# current step of 200 pA lasting 300 ms.
##################################


from __future__ import print_function
import numpy as np
import pyNN
from pyNN import *
import pyNN.nest as sim
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg') 
	 


if __name__=='__main__':


	#Set the simulation time step
	sim.setup(timestep=0.1, spike_precision="on_grid") #timestep= 0.1 ms

	# Set the neuron model class
	neuron_model = sim.EIF_cond_exp_isfa_ista # an Adaptive Exponential I&F Neuron

	#Set the paramaters for each cell
	# excitatory cell: regular-spiking (RS_cell)

	neuron_parameters_RS_cell = {
    	'cm': 0.150,  # nF - tot membrane capacitance
    	'v_reset': -65, # mV - reset after spike
    	'v_rest': -65,  # mV - resting potential E_leak
    	'v_thresh': -50, # mV - fixed spike threshold
    	'e_rev_I': -80, #
    	'e_rev_E': 0, #
    	'tau_m': 15, # ms - time constant of leak conductance (=cm/gl)
    	'tau_refrac': 5, # ms - refractory period
    	'tau_syn_E': 5., # ms
    	'tau_syn_I': 5., # ms
    	'a':4, # nS - conductance of adaptation variable
    	'b':0.02, # nA - increment to the adaptation variable 
    	'tau_w': 500, # ms - time constant of adaptation variable 
    	'delta_T': 2 } # mV - steepness of exponential approach to threshold

    # inhibitory cell: fast-spiking (FS_cell)
	neuron_parameters_FS_cell={
    	'cm': 0.150,
    	'v_reset': -65,
    	'v_rest': -65,
    	'v_thresh': -50,
    	'e_rev_I': -80,
    	'e_rev_E': 0,
    	'tau_m': 15,
    	'tau_refrac': 5,
    	'tau_syn_E': 5.,
    	'tau_syn_I': 5.,
    	'a':0,
    	'b':0,
    	'tau_w': 500,
    	'delta_T': 0.5 }


   	#Create  the two cell populations of 1 neuron 
 	Neurons_RS = sim.Population(size=1, cellclass=neuron_model, cellparams=neuron_parameters_RS_cell) 
	Neurons_FS = sim.Population(size=1, cellclass=neuron_model, cellparams=neuron_parameters_FS_cell)  
		
	#V value initialization
	sim.initialize(Neurons_RS,v=-65.0) # v:mV 
	sim.initialize(Neurons_FS,v=-65.0) # v:mV

	#parameters recorded : membrane potential & spikes
	Neurons_RS.record('v')
	Neurons_RS.record('spikes')

	Neurons_FS.record('v')
	Neurons_FS.record('spikes')

	#The external input 
	pulse=sim.DCSource(amplitude=0.2, start=100, stop=400) #amplitude=200pA, start=100ms and stop=400ms

	#representaton of the pulse injected
	x=[0,50,99,100,200,300,399,400,600]
	y=[0,0,0,200,200,200,200,0,0]
	fig=plt.figure()
	plt.plot(x,y)
	plt.xlabel("time(ms)")
	plt.ylabel("I(pA)")
	plt.ylim((0,210))
	plt.title('DC current source')
	plt.setp(plt.gca().get_xticklabels(),visible=True)
	fig.savefig('input.png')

	
	#Adding the external input to the cells before running the simulation 
	Neurons_RS.inject(pulse)
	Neurons_FS.inject(pulse)

	sim.run(600) #ms

	#getting the data recorded
	data_v_RS=Neurons_RS.get_data('v')
	data_s_RS=Neurons_RS.get_data('spikes')

	data_v_FS=Neurons_FS.get_data('v')
	data_s_FS=Neurons_FS.get_data('spikes')

	segment_RS=data_s_RS.segments[0]
	times_RS=data_v_RS.segments[0].analogsignals[0].times
	signal_RS= data_v_RS.segments[0].analogsignals[0]

	segment_FS=data_s_FS.segments[0]
	times_FS=data_v_FS.segments[0].analogsignals[0].times
	signal_FS= data_v_FS.segments[0].analogsignals[0]


	#helpers for plotting our data
	def plot_spiketrains(segment):
		for spiketrain in segment.spiketrains[0]:
			y=np.ones_like(spiketrain)*-30
			plt.plot(spiketrain,y,'.')
			plt.ylabel(segment.name)
			plt.setp(plt.gca().get_xticklabels(),visible=False)

	def plot_signal(signal,times,c):
		plt.plot(times,signal, color=c)
		plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
		plt.setp(plt.gca().get_xticklabels(),visible=False)
		plt.legend()


	#Plotting 
	#fig1 : RS_cell in green

	fig1=plt.figure()
	plot_signal(signal_RS,times_RS,'g')
	plot_spiketrains(segment_RS)

	plt.xlabel("time (%s)" % signal_RS.times.units._dimensionality.string)
	plt.setp(plt.gca().get_xticklabels(), visible=True)
	plt.ylim((-80,-20))
	plt.title('RS_cell')
	fig1.savefig('RS_cell.png')
	plt.close()
	

	#fig2 : FS_cell in red

	fig2=plt.figure()
	plot_signal(signal_FS,times_FS,'r')
	plot_spiketrains(segment_FS)

	plt.xlabel("time (%s)" % signal_FS.times.units._dimensionality.string)
	plt.setp(plt.gca().get_xticklabels(), visible=True)
	plt.ylim((-80,-20))
	plt.title('FS_cell')
	fig2.savefig('FS_cell.png')
	plt.close()


	sim.end()        



	