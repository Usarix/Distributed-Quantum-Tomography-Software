# Distributed-Quantum-Tomography-Software
Code to run distributed quantum tomography via MUBs (arXiv:2604.09775)

The code presented is used for the distributed tomography protocol, it requires Qiskit as a main dependency.

# Description
The program delivers the statistical properties of the distributed computing protocol via MUBs, it has different functions to make this procedure more friendly and editable, only the part after the functions is the code running with the given parameters. If you want to edit certain parts of the procedure, find the desire function and edit it. Hopefully i will separate the function part with the code part in the near future.


# How to use
The code is prepared in such manner that can be launched from the terminal given the rights arguments (argparse).

-q = The total amount of qubits for the simulation (its a narg argument, this means you can put more than just one value into the argument and it will be accepted)(It automatically does every partition possible for that amount in the case of non noise)

-s = The total shots for which the simulation will be done (narg)

-e = The amount of states to be done to have a good statistical analisis

-n = The noise of the simulation (This is automatically done just in the middle for bipartites to simplify)(default = 0 and if it is zero then run the non noise model)

-c = The amount of cores to the software to use. The default is 100 as it is the amount i have available, if not specified it will use all cores. For testing your hardware, use one core first for small amount of qubits and see how much time it takes

The results are given in the file "distributed_tomography_data.txt" and the data is easily readable for its analysis and use.
