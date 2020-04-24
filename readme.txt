Pynn demo files to simulate networks of RS-FS cells  
---------------------------------------------------


Those PyNN demo files simulate a network of regular-spiking (RS)
excitatory neurons and fast-spiking (FS) inhibitory neurons. We study
here the network at the level of single cells, then his spontaneous
activity and finally its response to time-varying external input, as
described in the following paper:


Zerlaut Y, Chemla S, Chavane F, Destexhe A (2018) Modeling mesoscopic
cortical dynamics using a mean-field model of conductance-based
networks of adaptive exponential integrate-and-fire neurons. J Comput
Neurosci 44:45-61

These files were contributed By Amelie Soler (Destexhe lab). Many
example output images are provided in the subfolders.

This paper presents a RS-FS mean-field model of networks of Adaptive
Exponential (AdEx) integrate-and-fire neurons, with conductance-based
synaptic interactions. It uses a Master Equation formalism, together
with a semi-analytic approach to the transfer function of AdEx neurons
to describe the average dynamics of the coupled populations. It
compares the predictions of this mean-field model to simulated
networks of RS-FS cells, first at the level of the spontaneous
activity of the network. Second, it investigates the response of the
network to time-varying external input. Finally, to model VSDi
signals, a one-dimensional ring model made of interconnected RS-FS
mean-field units is used.


The simulations shown here are reproductions of Figure 2 (simulation
1), Figure 3-a,b (simulation 2) and Figure 5-a,b (simulation 3) of the
paper.
 
SImulation 1: It shows the response of the single cell models (RS in
green and FS in red) to an external current step of 200 pA lasting 300
ms. The parameters of the cells are the same as the ones presented on
Table 1 in the article. The membrane potentials are plotted with the
spikes (represented with dots) on the same graph.


Simulation 2: This script shows the spontaneous activity of the
network. The network is made of 8000 excitatory neurons and 2000
inhibitory neurons. Those two populations of neurons are randomly
connected (internally and mutually) with a connectivity probability of
5%. Plus a feedforward input (a ramp of 4HZ) on the inhibitory
population and on the excitatory population is added. We simulate the
network for 1000ms. The spiking activity and firing rate of the
network is plotted (green: excitatory neurons, red: inhibitory
neurons).


Simulation 3: This last script shows the response of the network to
time-varying external input. The duration of simulation and the
construction of the network is the same. Only here, the feedforward
input on the inhibitory population is varying between 4Hz and 6H
(following theoretically a gaussian in the paper). During 100ms, it’s
the first part of the gaussian; here we simulate the gaussian used by
a ramp. Then for 200ms, it’s the second part of the gaussian, we
simulate it with a descending ramp. Identically, the spiking activity
and firing rate of the network is plotted (green: excitatory neurons,
red: inhibitory neurons).


If you use this for your research, please cite the above paper.

Amélie Soler
CNRS, Gif sur Yvette, France
http://cns.iaf.cnrs-gif.fr
