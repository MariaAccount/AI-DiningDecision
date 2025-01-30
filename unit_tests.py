import pytest
import json
import pandas as pd
from create_draw_tree import TreeNode

#Reads the json file and stores the data on the test class before any test being executed
@pytest.fixture(scope="class", autouse=True)
def load_data(request):
    with open('dados.json', 'r') as file:
        request.cls.test_data = json.load(file) 

#Garantees that load_data is executes before any teste inside this class
@pytest.mark.usefixtures("load_data")  
class Tests:
    #Runs the same test por several values
    @pytest.mark.parametrize("key_value", [0,1,2,3,4,5,6,7,8,9,10,11])
    def test_method(self, key_value):
        self.go_to_key(key_value)

    #Runs the test for a specific value
    def go_to_key(self, i):
        filepath = "restaurant_data.csv"
        examples = pd.read_csv(filepath, delimiter=' ', keep_default_na=False)
        attributes = examples.columns[1:-1]

        root = TreeNode(None, examples, None, 0)
        graph = root.create_graph(examples, attributes)

        assert self.compare(graph, self.test_data[i]) == self.test_data[i]["WillWait"]      
 
    #Compares the graph's branches with the results ("WillWait") on the json file
    def compare(self,graph,results):
        if graph.children == []:
            return graph.data         
        else:
            for child in graph.children: 
                if results[graph.data] == child.branch: 
                    return self.compare(child,results)                  
 
if __name__ == '__main__':
    unittest.main()

