
import ast
import pickle



if __name__ == '__main__':
    with open("output2", 'r') as file:
        lines = file.read().splitlines()

    GRAPH_PREFIX = "graph type: "
    DENSITY_PREFIX = "density : "
    SIZE_PREFIX = "size:  "

    graph = ""
    density = ""
    size = ""

    results = {}
    for line in lines:
        if line.startswith(GRAPH_PREFIX):
            graph = line[len(GRAPH_PREFIX):]
            results[graph] = {}
        if line.startswith(DENSITY_PREFIX):
            density = line[len(DENSITY_PREFIX):]
            results[graph][density] = {}
        if line.startswith(SIZE_PREFIX):
            size = line[len(SIZE_PREFIX):]
            results[graph][density][size] = {}
        if line.startswith("{"):
            data = ast.literal_eval(line)
            results[graph][density][size] = data

    with open("coa2.pickle", "wb") as file:
        pickle.dump(results,file)





















