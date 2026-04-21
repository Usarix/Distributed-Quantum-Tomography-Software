# Distributed-Quantum-Tomography-Software
Code to run distributed quantum tomography via MUBs (arXiv:2604.09775)

The code presented is used for the distributed tomography protocol, it requires Qiskit as a main dependency.

# How to use
The code is prepared in such manner that can be launched from the terminal given the rights arguments (argparse).
-q = The total amount of qubits that the simulation (its a narg argument, this means you can put more than just one value into the argument and it will be calculated)(It automatically does every partition possible for that amount)
-s = The total shots for which the simulation will be done (narg)
-e = The amount of states to be done to have a good statistical analisis
-n = The noise of the simulation (This is automatically done just in the middle for bipartites to simplify)(default = 0 and if it is zero then run the non noise model)
-c = The amount of cores to the software to use. The default is 100 as it is the amount i have available, if not specified it will use all cores. For testing your hardware, use one core first for small amount of qubits and see how much time it takes

# Trobuleshooting
If the program gives you an error about memory usage then the problem is in the parameters of cores and states (-c -e), to solve it, it is prefered that for great amounts of cores the amount of states is the same (cores=states) (This problem emerges because the qiskit circuits for all those states are saved in memory to make the software faster and the paralelization doesnt reference this memory, instead it creates new ones. Because i wasn't really in needing of solving this, i simply didnt)
