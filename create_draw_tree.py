import pandas as pd
import numpy as np
import pygraphviz as pgv
import json
import csv
import sys

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
        validate_examples(examples)
        graph = ID3(examples, "WillWait",atributes, self,None)
        return graph

#Generates an unique id for each Node
def generate_id():
    global unique_id
    unique_id += 1
    return unique_id

#Reads the examples from a CSV file separated by spaces
#keep_default_na is False because "None" is a valid value of the attribute Pat
def load_examples(filepath):
    return pd.read_csv(filepath,delimiter=' ', keep_default_na=False)

#Raises a ValueError if the examples can't be used to build a tree
def validate_examples(examples):
    columns = list(examples.columns)
    if len(columns) < 2 or columns[0] != "Example" or columns[-1] != "WillWait":
        raise ValueError("The first column must be 'Example' and the last column must be 'WillWait'")
    if examples.empty:
        raise ValueError("The dataset has no examples")
    if (examples.isna() | (examples == "")).any(axis=None):
        raise ValueError("The dataset has missing values")
    invalid_labels = set(examples["WillWait"]) - {"Yes", "No"}
    if invalid_labels:
        raise ValueError(f"WillWait must be 'Yes' or 'No', found: {sorted(invalid_labels)}")
    duplicated = examples["Example"][examples["Example"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicated examples: {duplicated}")

#Follows the branches of the tree according to the example, returning the result of the leaf
#The example can be a dictionary or a pandas Series
#Raises a ValueError if the example has no value for an attribute or a value never seen in the training data
def predict(tree, example):
    node = tree
    while node.children:
        attribute = node.data
        if attribute not in example:
            raise ValueError(f"The example has no value for the attribute '{attribute}'")
        value = example[attribute]
        for child in node.children:
            if child.branch == value:
                node = child
                break
        else:
            raise ValueError(f"Value '{value}' of the attribute '{attribute}' was not seen in the training data")
    return node.data

#Builds the tree and draws it. Invalid data ends the program with a clear message
def main(filepath="restaurant_data.csv"):
    try:
        examples = load_examples(filepath)
        attributes = examples.columns[1:-1]
        root = TreeNode(None,examples,None,0)
        graph = root.create_graph(examples, attributes)
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")

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
#If every attribute has zero gain, the first one is selected
def select_attribute(examples, attributes):
    gain = -1
    for at in attributes:
        if at != "Example":
            if information_gain(examples, at)> gain:
                gain = information_gain(examples, at)
                final_attribute = at           
    return final_attribute

#Creates a leaf with the result. If there is no tree yet, the leaf is the whole tree
def add_leaf(result, examples, root, branch, node_id):
    Node = TreeNode(result,examples["Example"],branch,node_id)
    if root.data is None:
        return Node
    root.add_child(Node)
    return root

#In the case there are no attributes, chooses the result depending if there are more Yes or No in the result
#A tie results in No
def chooseResult(examples,root,branch,node_id):
    if  examples.iloc[:, -1].tolist().count("Yes") > examples.iloc[:, -1].tolist().count("No"):
        return add_leaf('Yes',examples,root,branch,node_id)
    else:
        return add_leaf('No',examples,root,branch,node_id)

#Perfoms the algorithm ID3, returning a complete tree graph
def ID3(examples, target_attribute, attributes, root,branch):  
    node_id = generate_id()  
    if len(attributes) == 0:
        return chooseResult(examples,root,branch,node_id)
    elif all(WillWait == "Yes" for WillWait in examples.iloc[:, -1].tolist()):
        return add_leaf('Yes',examples,root,branch,node_id)
    elif all(WillWait == "No" for WillWait in examples.iloc[:, -1].tolist()):
        return add_leaf('No',examples,root,branch,node_id)
    else:
        A = select_attribute(examples, attributes)   
        if root.data != None:     
            Node = TreeNode(A, examples["Example"],branch,node_id)
            root.add_child(Node)
        else:
            Node = TreeNode(A, examples["Example"],branch,node_id)
            root = Node
        for vi, exs in examples.groupby(A):
            ID3(exs.drop(columns = A),target_attribute, attributes.drop(A), Node,vi)
    return root

#Calls the main function only when the file is run directly, not when it is imported
if __name__ == "__main__":
    main()