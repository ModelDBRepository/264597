#Tested with a 2.7 python version#
##################################
# This script shows the spontaneous
# activity of the network####


from __future__ import print_function
import numpy as np
import pyNN
from pyNN import *
import pyNN.nest as sim
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg') 
     


if __name__=='__main__':


    # size of the Population we will create 
    #(8000 exc neurons + 2000 inh neurons)
    N_exc = 8000 
    N_inh = 2000

    DT=0.1 #ms timestep
    tstop=1000 #ms simulation duration

    #Set the simulation time step
    sim.setup(timestep=DT, spike_precision="on_grid") 

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


    #Creation of  the two cell populations  
    Neurons_RS = sim.Population(size=N_exc, cellclass=neuron_model, cellparams=neuron_parameters_RS_cell) 
    Neurons_FS = sim.Population(size=N_inh, cellclass=neuron_model, cellparams=neuron_parameters_FS_cell) 

    #V value initialization
    sim.initialize(Neurons_RS,v=-65.0, gsyn_exc=0, gsyn_inh=0) # v:mV, gsyn_exc:nS, gsyn_inh:nS
    sim.initialize(Neurons_FS,v=-65.0, gsyn_exc=0, gsyn_inh=0) 


    ## RECURRENT CONNECTIONS
    #The two populations of neurons are randomly connected
    # (internally and mutually) with a connectivity probability of 5%

    # exc_exc
    exc_exc =sim.Projection(Neurons_RS, Neurons_RS,sim.FixedProbabilityConnector(0.05,allow_self_connections=False),receptor_type='excitatory',synapse_type=sim.StaticSynapse(weight=0.001))  
    # exc_inh
    exc_inh =sim.Projection(Neurons_RS, Neurons_FS,sim.FixedProbabilityConnector(0.05,allow_self_connections=False),receptor_type='excitatory',synapse_type=sim.StaticSynapse(weight=0.001))   
    # inh_exc
    inh_exc =sim.Projection(Neurons_FS, Neurons_RS,sim.FixedProbabilityConnector(0.05,allow_self_connections=False),receptor_type='inhibitory',synapse_type=sim.StaticSynapse(weight=0.005))   
    # inh_inh
    inh_inh =sim.Projection(Neurons_FS, Neurons_FS,sim.FixedProbabilityConnector(0.05,allow_self_connections=False),receptor_type='inhibitory',synapse_type=sim.StaticSynapse(weight=0.005))    


    ##FEEDFORWARD CONNECTIONS    
    # feedforward input on INH pop
    input_exc = sim.Population(N_exc, sim.SpikeSourcePoisson(rate=0)) 
    fdfrwd_to_exc=sim.Projection(input_exc,Neurons_RS,sim.FixedProbabilityConnector(0.05,allow_self_connections=False),synapse_type=sim.StaticSynapse(weight=0.001))
    # feedforward input on EXC pop
    fdfrwd_to_inh = sim.Projection(input_exc, Neurons_FS, sim.FixedProbabilityConnector(0.05,allow_self_connections=False),synapse_type=sim.StaticSynapse(weight=0.001))
    

    #To calcul the firing rate
    def rate(segment, bin_size):
        if segment == []:
            return NaN
        # crate bin edges based on the run_time (ms) and bin_size
        bin_edges=np.arange(0, tstop+bin_size, bin_size)
        hist = np.zeros(bin_edges.shape[0]-1)
        for spiketrains in segment.spiketrains:
            hist=hist+np.histogram(spiketrains,bin_edges)[0]
        return ((hist/ len (segment.spiketrains)) / bin_size)


    #Data to record
    Neurons_RS.record('spikes')
    Neurons_FS.record('spikes')


    class SetRate(object):

        #A callback which changes the firing rate of a population
        #of poisson processes at a fixed interval.

        def __init__(self, population, rate_generator, interval=0.5):
            assert isinstance(population.celltype, sim.SpikeSourcePoisson)
            self.population = population
            self.interval = interval
            self.rate_generator = rate_generator 

        def __call__(self, t):
            try:
                self.population.set(rate=next(rate_generator))
            except StopIteration:
                pass 
            return t+self.interval
            

    #To  generate the 4Hz ramp external drive that target
    # both the excitatory and inhibitory neurons:
    #every 0.5ms the firing rate is increased by 4/100 Hz
    #to reach 4Hz at 50ms of the simulation simuation
    interval=0.5
    l=[]
    k=0
    while k<=4:
        l.append(k)
        k=k+0.04
        

    rate_generator=iter(l)

    #Running the simulation
    sim.run(50,callbacks=[SetRate(input_exc,rate_generator,interval)]) #ms
    input_exc=sim.Population(N_exc,sim.SpikeSourcePoisson(rate=4))
    #after 50ms the external drive stays at 4Hz for the rest of the simulation    
    sim.run(tstop-50) #ms
    sim.end()    

    #getting the data recorded
    data_s_exc=Neurons_RS.get_data('spikes')
    data_s_inh=Neurons_FS.get_data('spikes')

    segment_s_exc=data_s_exc.segments[0]
    segment_s_inh=data_s_inh.segments[0]

    #To smooth out the firing rate values
    def moving_average(x,w):
        return np.convolve(x,np.ones(w),'valid')/ w

    PRe=rate(segment_s_exc,bin_size=0.1)*10**3  #Hz
    PRi=rate(segment_s_inh,bin_size=0.1)*10**3  #Hz

    #spiking activity sampled in 5ms (50*0.1 ms) time bins across the population
    fr_exc=moving_average(PRe,50)
    fr_inh=moving_average(PRi,50)

    m0=fr_exc[5000:10000].mean() #To calculate the mean we don't take
    m1=fr_inh[5000:10000].mean() #into account the transitional regime

    #Smoothing again our graph
    fr_exc=fr_exc[::10]
    fr_inh=fr_inh[::10]


    #plotting
    #1: spiking activity
    s=[]
    a=[]
    fig1=plt.figure(figsize=(10,6))
    for spiketrain in segment_s_exc.spiketrains:

        y=np.ones_like(spiketrain)*spiketrain.annotations['source_id']
        plt.plot(spiketrain,y,'.g')
        s.append(np.array(spiketrain))
        a.append(np.array(y))
    M_spikes_exc=[np.array(s), np.array(a)]
    

    s=[]
    a=[]
    for spiketrain in segment_s_inh.spiketrains:

        y=np.ones_like(spiketrain)*spiketrain.annotations['source_id']
        plt.plot(spiketrain,y,'.r')
        s.append(np.array(spiketrain))
        a.append(np.array(y))
    M_spikes_inh=[np.array(s), np.array(a)]

    plt.ylim((7595,8105))
    plt.xlabel('Time(ms)')
    plt.ylabel('Neuron index')
    plt.title('spiking activity - exc eurons (green), inh neurons (red)')
    fig1.savefig('spiking_activity.png')
    plt.close()

    #2: firing rate
    fig2=plt.figure(figsize=(10,6))
    plt.plot(fr_exc,'g')
    plt.plot(fr_inh,'r')
    plt.xlabel('time(ms)')
    plt.ylabel('v(Hz)')
    plt.title('mean firing rate exc(green):'+str(m0)+' sp/ms, inh(red):'+str(m1)+'sp/ms')
    fig2.savefig('firing_rate.png')
    plt.close()

    #representaton of the ext input 
    x=[0,10,20,30,40,50,100,300,400,600,800,1000]
    y=[0,0.8,1.6,2.4,3.2,4.0,4.0,4.0,4.0,4.0,4.0,4.0]
    fig=plt.figure()
    plt.plot(x,y)
    plt.xlabel("time(ms)")
    plt.ylabel("v(Hz)")
    plt.ylim((0,6))
    plt.title('External Input')
    plt.setp(plt.gca().get_xticklabels(),visible=True)
    fig.savefig('ext_input_spontaneous_activity.png')
