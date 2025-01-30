import pandas as pd
import numpy as np
import pygraphviz as pgv
import json
import csv

#Creates an unique id for each node in the graph
unique_id = 0

class TreeNode:
    #Inicializes node 
    #data is the name of the node, for example, Hun
    #value is the label of each example associated with the Node, for example, x1,x3,x6
    #branch is the label of the branch that is above the node
    #id is an unique id for the node
    def __init__(self, data,value,branch,id):
        self.data = data
        self.children = []
        self.value = value
        self.branch = branch
        self.id = id
    def add_child(self,child):
        self.children.append(child)
    
    #Draws the graph, using the library pygraphviz
    def draw_graph(self, graph=None, parent=None):
        if graph is None:
            graph = pgv.AGraph(directed=True)

        graph.add_node(self.id, label=self.data, value=self.value)
        if parent is not None:
            graph.add_edge(parent, self.id, label=self.branch)

        for child in self.children:
            child.draw_graph(graph, self.id)

        return graph
    #Creates a graph based on the class TreeNode
    def create_graph(self, examples,atributes):
        graph = ID3(examples, "WillWait",atributes, self,None)
        return graph

#Generates an unique id for each Node
def generate_id():
    global unique_id
    unique_id += 1
    return unique_id

def main():
    
    filepath = "restaurant_data.csv"
    examples = pd.read_csv(filepath,delimiter=' ', keep_default_na=False)
    attributes = examples.columns[1:-1]
    root = TreeNode(None,examples,None,0)
    graph = root.create_graph(examples, attributes)

    graph.draw_graph().draw("graph.png", prog="dot")

#Calculates entropy 
def calculate_entropy(column):
    content = column.value_counts(normalize=True)
    return -sum(content * np.log2(content))
  
#Calculates the information gain of an attribute
def information_gain(examples, attribute):
    gain_attribute = calculate_entropy(examples["WillWait"])
    groups = examples.groupby(attribute)
   
    for term, group in groups:
        proportion = len(group) / len(examples)
        group_entropy = calculate_entropy(group["WillWait"])
        
        gain_attribute -= proportion * group_entropy
    return round(gain_attribute, 10)

#Selects the next attribute, depending on the infomation gain of each one
def select_attribute(examples, attributes):
    gain = 0
    for at in attributes:
        if at != "Example":
            if information_gain(examples, at)> gain:
                gain = information_gain(examples, at)
                final_attribute = at           
    return final_attribute

#In the case there are no attributes, chooses the result depending if there are more Yes or No in the result
def chooseResult(examples,root):
    if  examples.iloc[:, -1].tolist().count("Yes") > examples.iloc[:, -1].tolist().count("No"):
        Node = TreeNode('Yes',examples["Example"],None)
        return root.add_child(Node)
    else:
        Node = TreeNode('No',examples["Example"],None)
        return root.add_child(Node)
    
#Perfoms the algorithm ID3, returning a complete tree graph
def ID3(examples, target_attribute, attributes, root,branch):  
    node_id = generate_id()  
    if len(attributes) == 0:
        return chooseResult(examples,root)
    elif all(WillWait == "Yes" for WillWait in examples.iloc[:, -1].tolist()):
        Node = TreeNode('Yes',examples["Example"],branch,node_id)
        return root.add_child(Node)
    elif all(WillWait == "No" for WillWait in examples.iloc[:, -1].tolist()):
        Node = TreeNode('No',examples["Example"],branch,node_id)
        return root.add_child(Node)
    else:
        A = select_attribute(examples, attributes)   
        if root.data != None:     
            Node = TreeNode(A, examples["Example"],branch,node_id)
            root.add_child(Node)
        else:
            Node = TreeNode(A, examples["Example"],branch,node_id)
            root = Node
        for vi, exs in examples.groupby(A):
            if exs.empty:
                chooseResult(examples, attributes,root)
            else:
                ID3(exs.drop(columns = A),target_attribute, attributes.drop(A), Node,vi)                
    return root

#Calls the main function
main()