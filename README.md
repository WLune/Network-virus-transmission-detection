main.py is for testing the general project, 
with no technical codes

propagation.py defines a class named 
SIRPropagation, by designating parameters, 
its instance can simulate the process of 
virus propagation, adjacent matrix and 
time-status matrix can be visited. 

infection_graph.py defines a class named
InfectionGraph, a networkx graph object
is wrapped in it, with several useful method.

source_detection.py defines several method
for source tracking, where each of them is
packed into a class.

A common procedure to run this project is:
1. create a virus propagation model with 
propagation.py. Remember to generate the 
underlying graph and call SIR() to complete
the propagation process.
2. use the propagation model to create an 
InfectionGraph object, and conduct certain
operation(s).
Specifically, passing the adjacent_matrix and
time-status[t] of propagation model as the 
parameters for InfectionGraph.
3. deploy specific source detection method.
In detail, create an object corresponding to
the method to be deployed, then, use the 
set_data() method to read an SourceGraph.
detect() method will return a list of tuples
that indicate the probability of each node to 
be the source.

