#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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



def complete_attendance(matrix_attendance):
    x1=matrix_attendance[0]
    for i in matrix_attendance[1:]:
        compare=np.logical_or(x1,i)
        x1=compare
    return compare


class Strategy():
    def __init__(self, memory_size, constant):
        self.weights=np.random.uniform(-1,1,memory_size)
        self.constant=constant
        self.errors=[]
        self.scores=[]
    def forecast(self,N,memory):
        predict_memory=np.sum(self.weights*memory) + self.constant*N
        
        return predict_memory


class Attendee(Agent):
    #TREE OF BEHAVIOR: TODO
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, nb_strategies):
        super().__init__(unique_id, model)
        
        self.memory=self.model.initial_history
        
        x=random.randint(0,self.model.grid.width)
        y=random.randint(0,self.model.grid.height)
        self.position=(x,y)
        #list of agent's actions
        self.attend=random.randint(0,1)
        self.prediction=0
        
        ######agent reporters##### 
        self.attendance=[]                                   #the attendance history
        self.personal_scores=[]#list of agent's success      #the success of decisions
        self.error_prediction=[]                             #the absolute error of predictions
        
        ######strategies#########
        self.strategies=[]
        for i in range(nb_strategies):
            self.strategies.append(Strategy(self.model.memory_size,self.model.constant))#random choice of nb_strategies from the strategy list
        
        self.strategy=random.choice(self.strategies)#pick an initial random strategy
    
    
    def distance_from_self(self,other):
        pass
        
    
    def find_friends(self):
        #find the neighbors in the graph not in the grid
        if nx.is_empty(self.model.G)==False:
            self.friends=self.model.G[self]
            self.friend_attendance=np.array([f.attendance for f in self.friends])
        else:
            return (None)
    
    def collect_friend_info(self):
        """version limited information"""
        
        #get the list of attendance for all neighbors
        list_attendance=[self.attendance]
        for n in self.friends:
            list_attendance.append(n.attendance)
            
        #get the global information on the attendance of the bar
        history= np.array(self.model.initial_history + self.model.datacollector.model_vars["Attendance"])#before the decison
        #check the attendance in the memory span of all the contacts
        attendance_neighbors=complete_attendance(list_attendance)#TODO Get the info for days where at least one person in network was there
        #only keep the information if contacts have been to the bar.
        self.memory=history*np.array(attendance_neighbors)[-self.model.memory_size:]#get history data for only those days
        
    def collect_global_info(self):
        """initial version"""
        history= np.array(self.model.initial_history + self.model.datacollector.model_vars["Attendance"])#before the decison
        self.memory=history[-self.model.memory_size:]#update personal history: could be empty
        self.friend_attendance=np.array(np.array([f.attendance for f in self.friends]))
    
    def predict_attendance(self):
        #based on the chosen strategy, predict attendance as sum of wieghts and history
        #if higher than a threshold, then stay_home if not go to bar.
        self.prediction=self.strategy.forecast(self.model.num_agents,self.memory)
        
    def predict_friend_attendance(self):
        """version social"""
        #predict friend attendance: use the same strategy
        self.prediction_friends=self.strategy.forecast(len(self.friends),np.sum(self.friend_attendance,0))
        
    def imitate_friend_strategy(self,sucess=False):
        #can decide to choose a strategy from one of the friends
        if success==False:
            #pick a random friend and add his strategy to yours
            best_friend_strategy=random.choice([f.strategy for f in self.friends])
            
            
        #pick the most successful friend if success==true
        if success==True:
            #make sur not to change the personal score of the agent or friend
            best_score=self.personal_score.copy()
            best_friend_strategy=None
            for friend in self.friends:
                fsocres=sum(friend.personal_scores[-self.memory_size:])
                if best_score<fscores:
                    best_friend_strategy=friend.strategy.copy()
                    best_score=friend.personal_score.copy()
        
        #add to strategies if not already one    
        if best_friend_strategy not in self.strategies:
            self.strategies.append(best_friend_strategy)
            self.strategy=self.strategies[-1]
    
    def mean_friend_prediction(self):
        #see where to add this in the scheduler!!!!
        return(math.mean([f.prediction for f in self.friends]))
        
    def attend_bar(self):
        #go to a patch corresponding to the bar: if in bar
        #add one to global attendance
        if self.prediction < self.model.threshold: 
            self.attend=1 
            #keep count of the time slots where the agent has attended the bar
            self.attendance.append(1)
            self.model.grid.move_agent(self,(14,14))
            
        else:
            self.attend=0
            self.attendance.append(0)

        
    def evaluate(self):
        self.model.grid.move_agent(self,self.position)
        attendance=compute_attendance(self.model)
        self.error_prediction.append(abs(self.prediction - attendance))#keep list of the prediction errors of the agent
        #add one to the score if the prediction was right: check if accessing datacollector is the fastest way to do it
        self.personal_scores.append(((attendance > self.model.threshold ) == (self.prediction > self.model.threshold)))
        
        best_strategy=self.strategy
        for st in self.strategies:
            prediction=st.forecast(self.model.num_agents,self.memory)
            error=abs(prediction-attendance)#make sure to forecast without the new attendance
            st.errors.append(error)#keep list of potential prediction errors of the strategies (had they been used)
            success= ((attendance > self.model.threshold ) == (prediction > self.model.threshold))#make sure the forecast was on the "right" side. 
            st.scores.append(success)
            
            if sum(st.errors[-self.model.memory_size:]) < sum(best_strategy.errors[-self.model.memory_size:]):
                self.strategy = st
        
        #check which strategy would have work better for a given memory size: sum of the sum of wights times historical attendance tested for a given number of days (the same memory sizer in the netlogo one) 
 
        
    def decide(self):
        self.collect_global_info()
        self.predict_attendance()
        self.attend_bar()
        

