#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mesa import Agent, Model
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector
 #to collect features during the #simulation
from mesa.space import MultiGrid
 #to generate the environment
#for computation and visualization purpose
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import networkx as nx
from itertools import combinations


# In[ ]:


def compute_attendance(model):
    return  sum([1 for a in model.schedule.agents if a.attend == 1])


# In[ ]:


def set_edges(g,t,p,seed=None):
    #keep option to set seed
    if seed is not None: 
        random.seed(seed) 
    #keep option to have other forms of networks
    if t=="random":
        for u, v in combinations(g,2):
            if random.random() < p:
                g.add_edge(u, v)


# In[ ]:


class EF_Model(Model):
    def __init__(self, N, threshold, memory_size, nb_strategies,width, height,type_network=None,param_network=None):
        self.num_agents = N
      
        self.G = nx.Graph()
        self.memory_size=memory_size
        self.threshold=threshold
        self.constant=0
        self.bar_location=(5,5)
        
        self.initial_history=[random.randint(0,self.num_agents) for i in range(self.memory_size)]
        self.grid = MultiGrid(width, height, True)
        self.schedule = StagedActivation(self,['evaluate','decide'])
        self.running = True 
         #set history n-values (memory-size * 2) [random 100]
       #check model reporters if they can be accessed by agents: dictionnary

        # Create agents
        
        for i in range(self.num_agents):
            #threshold=random threshold
            #memory=random memory size
            a = Attendee1(i, self, nb_strategies)
            self.schedule.add(a)
            
            #add the condition that agents start at home: in this model, as there are no visuals, it doesn't really matter where they are as going to the bar is only defined by self.attend
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))
            
            
            #Create the graph and links and neighborhood
            if type_network is not None:
                self.G.add_node(a)
        
        if type_network is not None:
            set_edges(self.G, type_network, param_network)
    
        
        
        
        #find the adjacent agents in the graph
        for ag in self.schedule.agents:
            ag.find_friends()
        
        self.datacollector = DataCollector(
            model_reporters = {"Attendance": compute_attendance})
        

            
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
        
        #once they have all decided whether to attend or not, then collect new data. 
        #self.schedule.set_personal_score()
        #self.schedule.update_strategy()



