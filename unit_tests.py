import pytest
import json
import pandas as pd
from pathlib import Path
from create_draw_tree import (
    TreeNode,
    calculate_entropy,
    information_gain,
    load_examples,
    main,
    predict,
    select_attribute,
)

#Builds a DataFrame of examples with the columns Example, A, B and WillWait
def make_examples(rows, columns=("Example", "A", "B", "WillWait")):
    return pd.DataFrame(rows, columns=list(columns))

#Builds the tree in the same way as the main function
def build_tree(examples):
    root = TreeNode(None, examples, None, 0)
    return root.create_graph(examples, examples.columns[1:-1])

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

class TestEntropy:
    @pytest.mark.parametrize("values, expected", [
        (["Yes", "Yes", "Yes"], 0),
        (["No"], 0),
        (["Yes", "No"], 1),
        (["Yes", "Yes", "Yes", "No"], 0.8112781245),
    ])
    def test_known_values(self, values, expected):
        assert calculate_entropy(pd.Series(values)) == pytest.approx(expected)

    def test_empty_column(self):
        assert calculate_entropy(pd.Series([], dtype=object)) == 0

class TestInformationGain:
    def test_attribute_that_separates_the_results(self):
        examples = make_examples([["x1", "a", "c", "Yes"], ["x2", "b", "c", "No"]])
        assert information_gain(examples, "A") == 1

    def test_attribute_with_a_single_value(self):
        examples = make_examples([["x1", "a", "c", "Yes"], ["x2", "b", "c", "No"]])
        assert information_gain(examples, "B") == 0

    def test_restaurant_data(self):
        examples = load_examples("restaurant_data.csv")
        assert information_gain(examples, "Pat") == pytest.approx(0.5408520829)
        assert information_gain(examples, "Type") == pytest.approx(0)

class TestSelectAttribute:
    def test_selects_the_highest_gain(self):
        examples = make_examples([["x1", "c", "a", "Yes"], ["x2", "c", "b", "No"]])
        assert select_attribute(examples, examples.columns[1:-1]) == "B"

    def test_ignores_the_example_column(self):
        examples = make_examples([["x1", "a", "c", "Yes"], ["x2", "b", "c", "No"]])
        assert select_attribute(examples, examples.columns[:-1]) == "A"

    def test_zero_gain_selects_the_first_attribute(self):
        examples = make_examples([["x1", "a", "c", "Yes"], ["x2", "a", "c", "No"]])
        assert select_attribute(examples, examples.columns[1:-1]) == "A"

class TestEdgeCases:
    @pytest.mark.parametrize("result", ["Yes", "No"])
    def test_every_example_has_the_same_result(self, result):
        tree = build_tree(make_examples([["x1", "a", "c", result], ["x2", "b", "d", result]]))
        assert tree.data == result
        assert tree.children == []

    def test_single_example(self):
        tree = build_tree(make_examples([["x1", "a", "c", "Yes"]]))
        assert tree.data == "Yes"

    def test_tree_with_only_a_leaf_can_be_drawn(self):
        tree = build_tree(make_examples([["x1", "a", "c", "Yes"]]))
        assert tree.draw_graph().number_of_nodes() == 1

    def test_xor_has_zero_gain_at_the_root_and_is_still_learned(self):
        examples = make_examples([
            ["x1", "0", "0", "No"],
            ["x2", "0", "1", "Yes"],
            ["x3", "1", "0", "Yes"],
            ["x4", "1", "1", "No"],
        ])
        tree = build_tree(examples)
        for _, row in examples.iterrows():
            assert predict(tree, row) == row["WillWait"]

    @pytest.mark.parametrize("results, expected", [
        (["Yes", "Yes", "No"], "Yes"),
        (["No", "No", "Yes"], "No"),
        (["Yes", "No"], "No"),
    ])
    def test_contradictory_examples_use_the_majority(self, results, expected):
        rows = [[f"x{i}", "a", "c", result] for i, result in enumerate(results)]
        examples = make_examples(rows)
        assert predict(build_tree(examples), examples.iloc[0]) == expected

    def test_contradiction_in_only_one_branch(self):
        examples = make_examples([
            ["x1", "p", "c", "Yes"],
            ["x2", "q", "c", "No"],
            ["x3", "q", "c", "No"],
            ["x4", "q", "c", "Yes"],
        ])
        tree = build_tree(examples)
        assert predict(tree, examples.iloc[0]) == "Yes"
        assert predict(tree, examples.iloc[1]) == "No"

    def test_no_attributes_uses_the_majority(self):
        columns = ("Example", "WillWait")
        tree = build_tree(make_examples([["x1", "Yes"], ["x2", "Yes"], ["x3", "No"]], columns))
        assert tree.data == "Yes"

    def test_identical_examples_with_different_ids(self):
        examples = make_examples([["x1", "a", "c", "Yes"], ["x2", "a", "c", "Yes"], ["x3", "b", "c", "No"]])
        tree = build_tree(examples)
        for _, row in examples.iterrows():
            assert predict(tree, row) == row["WillWait"]

    def test_none_is_a_valid_value(self, tmp_path):
        file = tmp_path / "data.csv"
        file.write_text("Example A WillWait\nx1 None No\nx2 Some Yes\n")
        examples = load_examples(file)
        assert predict(build_tree(examples), examples.iloc[0]) == "No"

class TestInvalidData:
    @pytest.mark.parametrize("label", ["Maybe", "yes", "NO", ""])
    def test_invalid_result(self, label):
        examples = make_examples([["x1", "a", "c", label], ["x2", "b", "d", "Yes"]])
        with pytest.raises(ValueError):
            build_tree(examples)

    def test_invalid_result_is_named_in_the_message(self):
        examples = make_examples([["x1", "a", "c", "Maybe"], ["x2", "b", "d", "Yes"]])
        with pytest.raises(ValueError, match="Maybe"):
            build_tree(examples)

    @pytest.mark.parametrize("columns", [
        ("Example", "A", "B"),
        ("A", "B", "WillWait"),
        ("Example", "WillWait", "A"),
    ])
    def test_wrong_columns(self, columns):
        examples = make_examples([["x1", "a", "Yes"], ["x2", "b", "No"]], columns)
        with pytest.raises(ValueError, match="column"):
            build_tree(examples)

    def test_no_columns(self):
        with pytest.raises(ValueError, match="column"):
            build_tree(pd.DataFrame())

    def test_no_examples(self):
        with pytest.raises(ValueError, match="no examples"):
            build_tree(make_examples([]))

    @pytest.mark.parametrize("missing", [None, float("nan"), ""])
    def test_missing_value(self, missing):
        examples = make_examples([["x1", missing, "c", "Yes"], ["x2", "b", "d", "No"]])
        with pytest.raises(ValueError, match="missing"):
            build_tree(examples)

    def test_duplicated_example_id(self):
        examples = make_examples([["x1", "a", "c", "Yes"], ["x1", "b", "d", "No"]])
        with pytest.raises(ValueError, match="x1"):
            build_tree(examples)

class TestLoadExamples:
    def write(self, tmp_path, content):
        file = tmp_path / "data.csv"
        file.write_text(content)
        return file

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_examples(tmp_path / "missing.csv")

    def test_empty_file(self, tmp_path):
        with pytest.raises(ValueError):
            load_examples(self.write(tmp_path, ""))

    def test_row_with_too_many_values(self, tmp_path):
        content = "Example A WillWait\nx1 a Yes\nx2 b c No\n"
        with pytest.raises(ValueError):
            load_examples(self.write(tmp_path, content))

    def test_row_with_too_few_values(self, tmp_path):
        examples = load_examples(self.write(tmp_path, "Example A WillWait\nx1 a Yes\nx2 b\n"))
        with pytest.raises(ValueError, match="missing"):
            build_tree(examples)

    def test_file_with_only_the_header(self, tmp_path):
        examples = load_examples(self.write(tmp_path, "Example A WillWait\n"))
        with pytest.raises(ValueError, match="no examples"):
            build_tree(examples)

class TestPredict:
    def tree(self):
        return build_tree(make_examples([["x1", "p", "c", "Yes"], ["x2", "q", "c", "No"]]))

    def test_dictionary(self):
        assert predict(self.tree(), {"A": "p", "B": "c"}) == "Yes"
        assert predict(self.tree(), {"A": "q", "B": "c"}) == "No"

    def test_series(self):
        assert predict(self.tree(), pd.Series({"A": "q", "B": "c"})) == "No"

    def test_restaurant_data(self):
        examples = load_examples("restaurant_data.csv")
        tree = build_tree(examples)
        for _, row in examples.iterrows():
            assert predict(tree, row) == row["WillWait"]

    def test_attributes_not_used_by_the_tree_are_not_needed(self):
        assert predict(self.tree(), {"A": "p"}) == "Yes"

    def test_tree_with_only_a_leaf(self):
        tree = build_tree(make_examples([["x1", "a", "c", "Yes"]]))
        assert predict(tree, {}) == "Yes"

    def test_value_not_seen_in_training(self):
        with pytest.raises(ValueError, match="'z' of the attribute 'A'"):
            predict(self.tree(), {"A": "z", "B": "c"})

    def test_missing_attribute(self):
        with pytest.raises(ValueError, match="attribute 'A'"):
            predict(self.tree(), {"B": "c"})

class TestMain:
    @pytest.mark.parametrize("content, message", [
        ("", "No columns"),
        ("Example A WillWait\n", "no examples"),
        ("Example A WillWait\nx1 a Maybe\nx2 b Yes\n", "Maybe"),
        ("Example A\nx1 a\n", "column"),
        ("Example A WillWait\nx1 a Yes\nx2 b\n", "missing"),
        ("Example A WillWait\nx1 a Yes\nx2 b c No\n", "Expected 3 fields"),
    ])
    def test_invalid_file_ends_with_a_clear_message(self, tmp_path, monkeypatch, content, message):
        monkeypatch.chdir(tmp_path)
        file = tmp_path / "data.csv"
        file.write_text(content)
        with pytest.raises(SystemExit) as error:
            main(file)
        assert error.value.code.startswith("Error: ")
        assert message in error.value.code
        assert not (tmp_path / "graph.png").exists()

    def test_missing_file_ends_with_a_clear_message(self, tmp_path):
        with pytest.raises(SystemExit) as error:
            main(tmp_path / "missing.csv")
        assert error.value.code.startswith("Error: ")
        assert "missing.csv" in error.value.code

    def test_valid_file_draws_the_graph(self, tmp_path, monkeypatch):
        filepath = Path("restaurant_data.csv").resolve()
        monkeypatch.chdir(tmp_path)
        main(filepath)
        assert (tmp_path / "graph.png").exists()

if __name__ == '__main__':
    unittest.main()

