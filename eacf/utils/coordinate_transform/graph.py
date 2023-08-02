def topological_sort(graph_unsorted):
    graph_sorted = []
    graph_unsorted = dict(graph_unsorted)

    while graph_unsorted:
        acyclic = False
        for node, edges in list(graph_unsorted.items()):
            for edge in edges:
                if edge in graph_unsorted:
                    break
            else:
                acyclic = True
                del graph_unsorted[node]
                graph_sorted.append((node, edges))

        if not acyclic:
            raise RuntimeError("A cyclic dependency occured.")

    return graph_sorted