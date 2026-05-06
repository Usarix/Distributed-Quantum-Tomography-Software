import numpy as np
import random
import math
import matplotlib.pyplot as plt
#import qiskit
from qiskit import QuantumCircuit 
from qiskit import transpile
from qiskit.visualization import plot_state_city 
from qiskit.quantum_info import random_statevector, Statevector, Operator
from qiskit_aer.primitives import Sampler
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as SamplerV1
from qiskit_aer.primitives import SamplerV2
from qiskit.result import QuasiDistribution
from qiskit.transpiler import CouplingMap

#import csv

from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import partial_trace
from qiskit.circuit.random import random_circuit
from qiskit.circuit.random import random_clifford_circuit

from qiskit.visualization import plot_state_city
#from LinearAlgebra import * 

#import seaborn as sns
import itertools
from itertools import product
import ast

import multiprocessing
from joblib import Parallel, delayed, dump, load

import argparse


# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


parser = argparse.ArgumentParser(description ='Do quantum state tomography for distributed quantum computing and obtain its statisitc')
parser.add_argument('-q', '--qubits',nargs="+", 
                    type = int,
                    help ='The total amount of qubits that the distributed procedure will simulate')

parser.add_argument('-s', '--shots', nargs="+",
                    type = int,
                    help ='Lists of the shots that will be receiving each experiment')
parser.add_argument('-e', '--states', 
                    type = int,
                    help ='Number of states that the tomography procedure will be done')
parser.add_argument('-n', '--noise',
                    type = float,
                    help = 'A float that represents the lambda noise of the system. If 0 then the program will use the non noise method. Default = 0',
                    default = 0)
parser.add_argument('-c', '--cores',
                    type = int,
                    help = "The amount of cores the code will be using (Recommendable that this is greater or equal to the number of states to avoid memory overload)",
                    default = 100)


args = parser.parse_args()
#print(args.accumulate(args.integers))





#https://arxiv.org/abs/2311.11698
def AlgoritmoCircuitosMubs(N):

    nQbits = N
    if nQbits < 2:
        ## CREAR EL CIRCUITO A MANO (NADA;H;H+S))
        #print('Qubits Amount cannot be less than two')
        ListaA = [[0,0],[1,1]]
        ListaB = [[0,[0,0, 0]],[1,[0,0, 0]]]
        
        return ListaA, ListaB
    #Paso 0: La creacion de la base binaria
    J = []   #Base binaria ordenada, ver paper
    BinariaMatriz = [list(bin(i).split('b')[1].zfill(nQbits)) for i in range(2**nQbits)]
    for ii in range(2**nQbits):
        BinariaMatriz[ii].reverse()
        J.append(BinariaMatriz[ii])
        J[ii] = list(map(int, J[ii]))

    #Paso 1
    # El ejemplo para 3 qubits
    
    p_x = [0]*(nQbits+1)    #El orden del polinomio es: p(x) = 1 + x + .. + x^nina

    #####   Aqui habria que colocar la manera de encontrar los valores de la tabla e introducirlos en p_x
    #Una idea que se me ocurre es con if gates pero hay que portear la tabla de alguna manera
    match nQbits:
        case 2:
            p_x = [1,1,1] #2
        case 3:
            p_x = [1,1,0,1]
        case 4:
            p_x = [1,1,0,0,1]
        case 5:
            p_x = [1,0,1,0,0,1]
        case 6:
            p_x = [1,1,0,0,0,0,1]
        case 7:
            p_x = [1,1,0,0,0,0,0,1]
        case 8:
            p_x = [1,1,0,1,1,0,0,0,1]
        case 9:
            p_x = [1,1,0,0,0,0,0,0,0,1]
        case 10:
            p_x = [1,0,0,1,0,0,0,0,0,0,1]
    #####


    ##Paso 2    ##Primer indice representa el vector x^(primer indice), el segundo indice es el componente de este vector
    XVectorMatrix = [0]*(2*nQbits-1)
    for ii in range(2*nQbits-1):
        XVectorMatrix[ii] = [0]*(nQbits)
    for jj in range(2*nQbits-1):
        if jj < (nQbits):
            for ii in range(nQbits):
                if ii == jj:
                    XVectorMatrix[jj][ii] = 1

        elif jj == (nQbits):
            for ii in range(nQbits):
                XVectorMatrix[jj][ii] = p_x[ii]

        elif jj > (nQbits):
            for ii in range(nQbits):
                if ii == 0:
                    XVectorMatrix[jj][ii] = (XVectorMatrix[jj-1][nQbits-1] * p_x[ii])
                else:
                    XVectorMatrix[jj][ii] = XVectorMatrix[jj-1][ii-1] + (XVectorMatrix[jj-1][nQbits-1] * p_x[ii])
    ##Fin paso 2

    ##Paso 3    ##La matriz es simetrica asi que da lo mismo si el primer indice es fila o columna
    M_0 = [0]*(nQbits)
    for ii in range(nQbits):
        M_0[ii] = [0]*(nQbits)

    M_1 = [0]*(nQbits)
    for ii in range(nQbits):
        M_1[ii] = [0]*(nQbits)

    for ii in range(nQbits):
        for jj in range(nQbits):
            M_0[ii][jj] = XVectorMatrix[jj+ii][0]
            M_1[ii][jj] = XVectorMatrix[jj+ii][1]


    #Paso 3
    MatrizVectoresPorM_0 = [0]*(2*nQbits-1)
    MatrizVectoresPorM_1 = [0]*(2*nQbits-1)

    for ii in range(2*nQbits-1):
        matriz = np.matrix(M_0)
        vectortranspuesto = np.array(XVectorMatrix[ii]).T
        MatrizVectoresPorM_0[ii] = matriz.dot(vectortranspuesto)

        matriz = np.matrix(M_1)
        vectortranspuesto = np.array(XVectorMatrix[ii]).T
        MatrizVectoresPorM_1[ii] = matriz.dot(vectortranspuesto)


    #Paso Final

    tablaSolucionesA = [[0,2],[3,1]]

    listaA = []
    listaB = []
    grupos = []
    for ii in range(nQbits):
        firstloop = True
        for jj in range(ii, nQbits):
            grupos.append([ii,jj])
    #print(J[0])

    for jj in range(0,2**nQbits):
        arraydelosA = []    #Solo existe para ordenar mejor los a
        arraydelosB = []
        #listaA.append(jj)
        #listaB.append(jj)
        arraydelosA.append(jj)
        arraydelosB.append(jj)
        for ii in range(nQbits):
            a = [int(np.dot(np.array(J[jj]).T,np.array(MatrizVectoresPorM_0[2*ii])[0]))%2, int(np.dot(np.array(J[jj]).T,np.array(MatrizVectoresPorM_1[2*ii])[0]))%2]
            arraydelosA.append(tablaSolucionesA[a[0]][a[1]])
            #listaA.append(tablaSolucionesA[a[0]][a[1]])
        for tt in range(nQbits):
            for yy in range(tt, nQbits):
                if (tt != yy):

                    b = np.dot(np.array(J[jj]).T,np.array(MatrizVectoresPorM_0[tt+yy])[0])
                    arraydelosB.append([tt,yy,int(b%2)])   ###EL % no estaba originalmente, revisar en el paper si debe ser mod 2 (binario)
                    ##listaB.append([tt,yy,int(b)])
        listaA.append(arraydelosA)
        listaB.append(arraydelosB)
    return listaA, listaB

#ListaA => [j1,a0,a1, ... , an],[j2,....],....
#ListaB => [j1,[tt,yy, 1or0]],[j2,[tt,yy, 1or0]],...    Si es 1, hay enlace entre tt , yy. Si es 0 no hay enlace



#cB its the classical bit part of the creation of the circuit
def CrearCircuito(A, B, N, j,startQubit=0,limpiando = False,classicalB=0):
    if (limpiando == True):
        Circuito = QuantumCircuit(N, classicalB)
        #if (N == 1):
            #startQubit = 0
    else:
        Circuito = QuantumCircuit(QubitsN,classicalB)
    #Circuito.h(range(N))
    if (j == 0):
        None
    elif (j == 1):
        Circuito.h(range(startQubit,N+startQubit))
    else:
        j = j-1
        Circuito.h(range(startQubit,N+startQubit))
        for Qubit,kk in enumerate(A[j][1:]):
            for _ in range(kk):
                Circuito.s(Qubit+startQubit)
        if (N!=1):
            numeroDeCombinaciones = int(math.factorial(N)/(math.factorial(2)*math.factorial(N-2)))     #Para entrelazamientos
            for ii in range(numeroDeCombinaciones):
                if B[j][ii+1][2] == 1:
                    Circuito.cz(B[j][ii+1][0]+startQubit,B[j][ii+1][1]+startQubit)
    return Circuito

def count_gates(qc: QuantumCircuit):
    gate_count = { qubit: 0 for qubit in qc.qubits }
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count

def remove_idle_wires(qc: QuantumCircuit):
    qc_out = qc.copy()
    gate_count = count_gates(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
    return qc_out

def listaBetweenNumbers(start, length):
    id_arr = list(range(start, start+length))
    return id_arr

def ciclo_derecha(N):
    arr = np.arange(N + 1)  # crea [0, 1, 2, ..., N]
    # ciclar los elementos desde el segundo hasta el último
    arr[1:] = np.roll(arr[1:], 1)
    return arr


#From this point the algorithm will be put to test, before this is all the part respect to the algorithm

def standardMUBTomography(probs,qubitsAmount,basis, startingQubit = 0):
    #This is the standard mub tomography (obtained from anibal's thesis)
    density = 0
    #Base aa, qubit/vector ii
    for aa in range(2**qubitsAmount+1):
        mub = basis[aa]
        #print(mub)
        mubMatrix = Operator(mub).to_matrix()
        for ii in range(2**qubitsAmount):
            projector = np.outer(mubMatrix[:,ii],mubMatrix[:,ii].T.conj())
            if (type(probs[aa].get(ii)) == float):
                density = density + probs[aa].get(ii)*(projector - (1/(2**qubitsAmount+1))*Operator(mubs[0]).to_matrix())
    #print(Operator(basis[0]).to_matrix())
    #print(basis[1])
    return density

def TomographyMUBS(bas, listadeValores,probabilidades):
    density = 0
    idx = 0
    f = bas[0]
    s = bas[1]
    d_1 = 2**listadeValores[1]   ##Esto son lo que equivale a 2**qubitsn de arriba
    d_2 = 2**listadeValores[2]
    probs_mubs = [ prob.binary_probabilities() for prob in probabilidades ]

    for qq in range(d_1+1):
        first = f[qq]
        firstMatrix = Operator(first).to_matrix()
        for ww in range(d_2+1):
            prob = probs_mubs[ idx ]
            second = s[ww]
            secondMatrix = Operator(second).to_matrix()
            times_of_below_loop = 0
            for rr in range(d_2):
                projector_2 = np.outer(secondMatrix[:,rr],secondMatrix[:,rr].T.conj())
                for ee in range(d_1):
                    projector_1 = np.outer(firstMatrix[:,ee],firstMatrix[:,ee].T.conj())
                    bit_str = bin(times_of_below_loop)[2:].zfill(QubitsN)
                    if bit_str in prob:

                        
                        density += prob[bit_str]*np.kron(projector_2 - (Operator(s[0]).to_matrix())/(d_2+1),projector_1 - (Operator(f[0]).to_matrix())/(d_1+1))   
            
                    #else:
                        #projectores_debug.append(projector_1 - (Operator(f[0]).to_matrix())/(d_1+1))


                    
                    times_of_below_loop += 1
            
            idx += 1

    return density




def DensityResonstructionMSystemsMubs(basis, listadeValores, probs):

    B = []
    density = 0

    ds = []
    list_of_bases = []
    bases_mult = 1
    dim_mult = 1
    bas_dim_mults = []

    for ee in range(0,len(listadeValores)-1):
        ds.append(2**listadeValores[ee+1])
        list_of_bases.append(ds[ee]+1)

        bases_mult = bases_mult*(ds[ee]+1)
        dim_mult = dim_mult*ds[ee]

        bas_dim_mults.append((ds[ee]+1)*ds[ee])
    
    #d_1 = 2**listadeValores[1]
    #d_2 = 2**listadeValores[2]
    
    #bases_mult = (d_1+1)*(d_2+1)
    #dim_mult = (d_1)*(d_2)
    #bas_dim_mult_1 = (d_1+1)*(d_1)
    #bas_dim_mult_2 = (d_2+1)*(d_2)

    #c_matrix = np.zeros((bases_mult,dim_mult))

    prob_mubs = [ prob.binary_probabilities() for prob in probs ]


    n = listadeValores[1:]
    d = np.array(ds)
    m = np.array(list_of_bases)

    

    c = np.zeros(  (np.prod(m), *d)  )
    # print( c.shape )
     
    for bb in range(np.prod(m)):
        for kk in prob_mubs[bb]:
            prob = prob_mubs[bb][kk]
            kk_split = []
            qq_acumulado = 0
            for qq in reversed(n):
                kk_split.append( kk[qq_acumulado:qq_acumulado+qq] )
                qq_acumulado += qq
            kk_split = kk_split[::-1]
     
            idx_kk = [ bb ]
            for kk in kk_split:
                idx_kk.append( int(kk,2) )
            c[tuple(idx_kk)] = prob
    
 
    list_idx = [  ]
    for nn in range(len(n)):
        list_idx.append( nn )
        list_idx.append( nn+len(n) )
     
    c = c.reshape( *m, *d ).transpose(list_idx).reshape(m*d)



    #######################

    
    #for qq in range(bases_mult):
    #    probability = probs_mubs[qq]
    #    for kk in probability:
            



    
    #probs_mubs = probs
    #for qq in (range(bases_mult)):
    #    for ww in (range(dim_mult)):
    #        c_matrix[qq][ww] = probs_mubs[qq].get(ww,0)
    #print(ciclo_derecha(2*(len(ValoresSubQubits)-1)-1))
    #print((c_matrix.reshape(np.concatenate((np.array(list_of_bases),np.array(ds))))).shape)
    #c_matrix = c_matrix.reshape(np.concatenate((np.array(list_of_bases),np.array(ds)))).transpose(ciclo_derecha(2*(len(ValoresSubQubits)-1)-1)).reshape(bas_dim_mults)

    #c_matrix = c_matrix.reshape([d_1+1,d_2+1,d_1,d_2]).transpose([0,3,1,2]).reshape([bas_dim_mult_1,bas_dim_mult_2])
    
    
    #c = c_matrix
    #print(c.shape)

    debug_xd = []
    flag = True
    
    idx = 0
    for ss in range(1,len(listadeValores)):
        b_pack = []
        #c_pack = []
        base_ind = basis[ss-1]
        for qq in range(2**listadeValores[ss]+1):
            prob = prob_mubs[ idx ]
            idx += 1 
            mub = base_ind[qq]
            mubMatrix = Operator(mub).to_matrix()
            #c_m=[]
            b_m=[]
            times_of_below_loop = 0
            for ww in range(2**listadeValores[ss]):
                bit_str = bin(times_of_below_loop)[2:].zfill(sum(listadeValores))
                projector = np.outer(mubMatrix[:,ww],mubMatrix[:,ww].T.conj())
                
                if bit_str in prob:
                    b_pack.append((projector - (1/(2**listadeValores[ss]+1))*np.eye(2**(listadeValores[ss]))))
                else:
                    b_pack.append((projector - (1/(2**listadeValores[ss]+1))*np.eye(2**(listadeValores[ss]))))
                times_of_below_loop += 1

        B.append(np.array(b_pack))






    
    #B = np.array(B)
    #B = B.T
    #B[0] = B[0][::-1]
    #B[1] = B[1][::-1]
    B=B[::-1]
    #print(c)
    #c = c.T
    c = np.transpose(c, axes=range(c.ndim-1, -1, -1))
    #print(c)

    

    #density = LinearCombinationMatrices( c, B )   #This function doesnt work properly, use the sqr_matrix_combinations
    density = sqr_matrix_combination(c,B)
    
    #density = Process2Choi(density)
    
    return density

def sqr_matrix_combination(c, B):
    """
    Computes:
        Z = sum_{m1,...,mS} c[m1,...,mS] kron(B_{m1},...,B_{mS})

    Parameters
    ----------
    c : ndarray
        S-dimensional coefficient array
    B : list of ndarray
        Each element must be (R, R, M) or (R, C, M)

    Returns
    -------
    Z : ndarray
        Resulting square matrix
    other : dict
        Metadata
    """

    if not isinstance(B, list):
        raise ValueError("B must be a list of arrays (one per subsystem).")

    S = len(B)

    R = []
    C = []
    M = []
    processed = []

    for Bs in B:

        if Bs.ndim == 2:
            #MAL
            m, r = Bs.shape
            cdim = 1
            Bs = Bs.reshape(m, r, 1)

        elif Bs.ndim == 3:
            m, r, cdim = Bs.shape

        else:
            raise ValueError("Each subsystem must be 2D or 3D array")

        R.append(r)
        C.append(cdim)
        M.append(m)

        mats = []
        for k in range(m):
            W = Bs[k,:, :]
            if r != cdim:
                W = W @ W.T
            mats.append(W)

        processed.append(mats)

    total_dim = np.prod(R)
    Z = np.zeros((total_dim, total_dim), dtype=complex)

    #print(M)
    for indices in product(*[range(m) for m in M]):
        #print(indices)
        coeff = c[indices]

        kron_term = processed[0][indices[0]]
        for s in range(1, S):
            kron_term = np.kron(kron_term, processed[s][indices[s]])

        Z += coeff * kron_term

    other = {
        "Subsystems": S,
        "Operators": M,
        "Rows": R,
        "Columns": C,
        "WasItNonSquare": [R[i] != C[i] for i in range(S)],
        "Configuration": 5,
    }

    return Z

#%%time
#QubitsN = 5

def loop_rec(bases,n,mt):
    if n>= 1:
        for x in range(len(bases[(len(ValoresSubQubits)-1)-n])):
            mt[n-1] = x
            
            loop_rec(bases,n-1,mt)
    else:
        for qq in range(len(ValoresSubQubits)-1):
            if qq == 0:
                #print(mt)
                Sternengesang = bases[-1][mt[qq]]
            else:
                Sternengesang = Sternengesang & bases[-qq-1][mt[qq]]
        composedBases.append(Sternengesang)

###########################EDITAR ESTA CELDA PARA DEIFERENCIAR ENTRE SUBISTEMAS

def InputmubforCompose(circuito, basesmub):
    #basesmub = basesmub[::-1]
    counterofcircuit = 0
    
    for qq in range(len(basesmub)):
        #print(listaBetweenNumbers(ValoresSubQubits[0],ValoresSubQubits[qq+1]))
        #print(circuito)
        #print(basesmub[-1])
        circuito.compose(basesmub[qq].inverse(),listaBetweenNumbers(counterofcircuit,ValoresSubQubits[qq+1]), inplace=True )
        counterofcircuit += ValoresSubQubits[qq+1]

def haar_random_unitary(n):
    """Genera una matriz unitaria n x n distribuida según Haar."""
    X = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    Q = Q * ph
    return Q
def add_random_haar_gates(qc):
    """
    Aplica una unitaria Haar aleatoria a cada qubit usando qc.unitary().
    """
    for q in range(qc.num_qubits):
        U = haar_random_unitary(2)
        qc.unitary(U, [q], label="Haar")



def GenerateHaar(n):
    psi = np.random.randn(2**n) + 1j*np.random.randn(2**n)
    psi = psi / np.linalg.norm(psi)

    qc_rand_haar = QuantumCircuit(n)
    qc_rand_haar.initialize(psi, range(n))
    return qc_rand_haar



def GenerateRandomCircuit(depth, input_circuit ,random=True, noise = False):
    tomo = []
    
    if random == True:
        qc_random = input_circuit
        #add_random_haar_gates(qc_random) 
    else:
        qc_random = QuantumCircuit(QubitsN)
        qc_random.h(0)
        for qq in range(QubitsN-1):
            qc_random.cx(qq,qq+1)

        add_random_haar_gates(qc_random)

    if len(ValoresSubQubits) == 2:
        for ii in range(2**QubitsN+1):
            qc = QuantumCircuit(QubitsN)
            qc.compose(qc_random, inplace=True)
            qc.barrier()
            qc.compose(mubs[ii].inverse(),inplace = True)
            qc.measure_all()
            tomo.append( qc )
    else:


        cantidad = []
        for qq in range(len(ValoresSubQubits)-1):
            cantidad.append(BasesLimpias[qq])
        
        # Ahora usamos itertools.product con desempaquetado
        for mub_n in itertools.product(*cantidad):
            qc = QuantumCircuit(QubitsN)
            qc.compose(qc_random, inplace=True)
            qc.barrier()

            InputmubforCompose(qc, mub_n)
            
            qc.measure_all()
            tomo.append( qc )
            
        
        #for mub_0 in BasesLimpias[0]:
            #for mub_1 in BasesLimpias[1]:
                #qc = QuantumCircuit(QubitsN)
                #qc.compose(qc_random, inplace=True)
                #qc.barrier()
                #qc.compose( mub_0.inverse(),listaBetweenNumbers(ValoresSubQubits[0],ValoresSubQubits[1]), inplace=True )
                #qc.compose( mub_1.inverse(),listaBetweenNumbers(ValoresSubQubits[1],ValoresSubQubits[2]), inplace=True )
                #qc.measure_all()
                #tomo.append( qc )

    

                
    return tomo, qc, qc_random

#tomo[-1].draw('mpl',fold=-1)


###This is the algorithm featured in the Maximum Likelihood, Minimum Effort paper (https://arxiv.org/pdf/1106.5458)
def ProyectForPositiveEigenValues(matrix):
    values, vectors = np.linalg.eigh(matrix)

    idx = values.argsort()
    values = values[idx[::-1]]
    vectors = vectors.T
    vectors = vectors[idx[::-1]]
    #values[::-1].sort()
    a = 0
    d = len(values)-1
    i = d
    while i >= 0:
        if (values[i] + a/(i+1)) > 0:
            break
        else:
            a += values[i]
            values[i] = 0
            i -= 1
    for jj in range(i+1):
        values[jj] = values[jj] + a/(i+1)
    density = 0
    for qq in range(i+1):
        density += values[qq]*np.outer(vectors[qq], vectors[qq].conj().T)
    return density



def TraceDistance(density1, density2):
    matrix = density1 - density2
    eigva, eigve = np.linalg.eig(matrix)
    auto = 0
    for qq in range(len(eigva)):
        auto += np.abs(eigva[qq])
    distance = 0.5*auto
    return distance


def NegativityOfDensityMatrix(rho, qubits, small):

    #display(array_to_latex(rho))
    rho = rho.partial_transpose(qubits)
    #display(array_to_latex(rho))
    norm = np.linalg.norm(rho, ord="nuc")
    #negativity = (norm - 1)/(2**small-1)
    negativity = (norm - 1)/(2)

    
    return negativity

def ParallelStates(qq):
    #profundidad = 3*QubitsN


    if lambda_noise != 0:
        tomo, qc, random_qc = GenerateRandomCircuit(profundidad, circuitos_r[qq],False)
    else:
        tomo, qc, random_qc = GenerateRandomCircuit(profundidad, circuitos_r[qq], True)
   
    #print(tomo[-1])
    
    circ_density=random_qc.copy()
    circ_density.save_density_matrix()

    techo = math.ceil(QubitsN/2)
    suelo = math.floor(QubitsN/2)
    if techo == suelo:
        suelo = suelo - 1
    #print(suelo,techo)
    
     
    if lambda_noise != 0:

        noise_model = NoiseModel()
        error = depolarizing_error(lambda_noise, 2)
    
        #print(suelo, techo)
        noise_model.add_quantum_error(error, ["cx"], [suelo,techo])
        noise_model.add_quantum_error(error, ["cx"], [techo,suelo])

        #No noise
        #sampler = Sampler()
        #job = sampler.run( tomo , shots=tiros/forshots )
        #probs_dicts = job.result().quasi_dists
        #############
        #Noise
        backend = AerSimulator(noise_model=noise_model)

        #sampler_noise = SamplerV1(backend_options=dict(noise_model=noise_model))  
        #qasm = AerSimulator(
        #    method='density_matrix',
        #    noise_model=noise_model
        #)
        #circ_density = tomo.copy()
        #circ_density.save_density_matrix()
        #rho = np.array(
        #    qasm.run([tomo], shots=tiros/forshots).result().data()['density_matrix']
        #)

        #result_noise = sampler_noise.run(tomo, shots=tiros/forshots)
        #probs_discts = result_noise.quasi_dists
    
        connectivity = []

        #conecciones nodo 1
        for n in range(techo):
            for m in range(n+1,techo):
                connectivity.append( (n,m) )
                connectivity.append((m,n))
        #coneccion remota
        connectivity.append( (techo-1,techo))
        connectivity.append( (techo, techo-1))
        #coneccion nodo 2
        for n in range(techo, QubitsN):
            for m in range(n+1,QubitsN):
                connectivity.append( (n,m) )
                connectivity.append( (m,n) )

        coupling_map = CouplingMap( connectivity )
    
        tomo = transpile(tomo, coupling_map=coupling_map,basis_gates=noise_model.basis_gates, initial_layout=list(range(QubitsN)))
        #tomo = transpile(tomo, basis_gates=noise_model.basis_gates)
        sampler = SamplerV1(backend_options=dict(noise_model=noise_model))
        job = sampler.run(tomo, shots=tiros/forshots)
        probs_dicts = job.result().quasi_dists

        ######


    
        #print(len(probs_dicts))
        #print(BasesLimpias,probs_dicts)
        #if (len(ValoresSubQubits) == 2):
        #    densityMatrix = standardMUBTomography(probs_dicts, QubitsN, mubs)
        #else:
            #densityMatrix = TomographyMUBS(BasesLimpias,ValoresSubQubits,probs_dicts)
        densityMatrix = DensityResonstructionMSystemsMubs(BasesLimpias,ValoresSubQubits,probs_dicts)

        #print(ValoresSubQubits)
    
        densityMatrix = ProyectForPositiveEigenValues(densityMatrix)
        state = np.reshape(densityMatrix,(2**QubitsN,2**QubitsN))
        densityMatrix = DensityMatrix(densityMatrix)
        state = DensityMatrix(state)
    
        densityMatrixTheory_statevector = Statevector(random_qc)
        densityMatrixTheory = DensityMatrix(densityMatrixTheory_statevector)
        #print(densityMatrixTheory)

        #### Densidad matriz con ruido
        #qc.measure_all()
        sampler_noise = SamplerV2( options=dict(backend_options=dict(noise_model=noise_model) ) )
        qasm = AerSimulator(method='density_matrix',noise_model=noise_model)
    
        #circ_density=random_qc.copy()
        #circ_density.save_density_matrix()
        densityMatrixTheory = DensityMatrix( qasm.run(circ_density).result().data()['density_matrix'] )
        #densityMatrixTheory = DensityMatrix(densityMatrixTheory) 
        #densityMatrixTheory_statevector = Statevector(densityMatrixTheory)

        #print(densityMatrixTheory)

    else:
        sampler = Sampler()
        job = sampler.run( tomo , shots=tiros/forshots )
        probs_dicts = job.result().quasi_dists

        densityMatrix = DensityResonstructionMSystemsMubs(BasesLimpias,ValoresSubQubits,probs_dicts)
        densityMatrix = ProyectForPositiveEigenValues(densityMatrix)
        state = np.reshape(densityMatrix,(2**QubitsN,2**QubitsN))
        densityMatrix = DensityMatrix(densityMatrix)
        state = DensityMatrix(state)

        densityMatrixTheory_statevector = Statevector(random_qc)
        densityMatrixTheory = DensityMatrix(densityMatrixTheory_statevector)

    #print(tomo[-1])
    
    listForPartialTranspose = list(range(techo, sum(ValoresSubQubits)))
    
    #print(listForPartialTranspose, ValoresSubQubits, qc)
    tn = (NegativityOfDensityMatrix(densityMatrixTheory,listForPartialTranspose, ValoresSubQubits[-1]))
    n = (NegativityOfDensityMatrix(densityMatrix,listForPartialTranspose, ValoresSubQubits[-1])) 
    ##Reemplaza el theory por el theory_state
    F = (state_fidelity(state, densityMatrixTheory, validate=False))
    e2 = (np.linalg.norm(np.subtract(densityMatrixTheory,state),2)**2)
    enu = (np.linalg.norm(np.subtract(densityMatrixTheory, state), ord="nuc")**2)   #CUADRADO, SI VAS A ACAMBIARLO ACTUALIZA ESTA LINEA Y COMENTARIO
    return tn,n,F,e2,enu
    
# BASIC GATES
I = np.array([[1,0],[0,1]], dtype=complex)
H = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)


# Build the 24 single-qubit clifford matrices
CLIFFORDS = [
    np.linalg.matrix_power(S, a)
    @ np.linalg.matrix_power(H @ S, b)
    @ np.linalg.matrix_power(H, c)
    for a in range(4)
    for b in range(3)
    for c in range(2)
]


def random_clifford(rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    return CLIFFORDS[rng.integers(0, 24)]

def createCircuit(): #This function only exist to avoid the memory usage that happens after paralelization
    circuit = QuantumCircuit(QubitsN)
    
    return circuit


stored_sistem_distribution = []
stored_fidelities_mean = []
stored_std = []
stored_2norm = []
stored_nucnorm = []
stored_total_qubits = []
stored_shots = []
stored_projected_negativity = []
stored_theory_negativity = []
stored_negativity = []
stored_negativity_std = []
stored_shots = []

def integer_partitions(n, max_value=None):
    if max_value is None:
        max_value = n
        
    if n == 0:
        return [[]]
    
    partitions = []
    
    for i in range(min(max_value, n), 0, -1):
        for p in integer_partitions(n-i, i):
            partitions.append([i] + p)
            
    return partitions


def Partition(numbers, just_3_cases = False):
    all_partitions = []
    
    for n in numbers:
        if just_3_cases:
            all_partitions.append([n])

            if n> 2:
                a = (n+1)//2
                b = n//2
                if a!= b:
                    all_partitions.append([a,b])
                else:
                    all_partitions.append([a,a])
            all_partitions.append([1]*n)
        else:
            all_partitions.extend(integer_partitions(n))
        
    return all_partitions


qubitsCombinations = Partition(args.qubits, False)   ##True es el caso especial (Completa,mitad,full distribuida), False para que sean TODAS las particiones

#qubitsCombinations = [terminos for terminos in qubitsCombinations if len(terminos) == 2]


#qubitsCombinations = args.qubits

#print(qubitsCombinations)



for qq in range(len(qubitsCombinations)):
    qubitsCombinations[qq].sort(reverse=True)
    qubitsCombinations[qq].insert(0,0)



shotsperWholeSystem = args.shots
NumberOfStates = args.states
lambda_noise = args.noise
cores = args.cores

print(qubitsCombinations,shotsperWholeSystem,NumberOfStates)


matrix_nucnorm = np.zeros((len(qubitsCombinations), len(shotsperWholeSystem))) #Trace Norm
matrix_infnorm = np.zeros((len(qubitsCombinations), len(shotsperWholeSystem))) #2 norm/Operator
matrix_nuc_std = np.zeros((len(qubitsCombinations), len(shotsperWholeSystem)))
matrix_inf_std = np.zeros((len(qubitsCombinations), len(shotsperWholeSystem)))


#gates = [('h',1),('s',1),('x',1),('y',1),('z',1),('sdg',1),('cx',2),('cz',2),('swap',2)]
#gates_2 = ['cx','cz','swap']


for bb in range(len(qubitsCombinations)):
    ValoresSubQubits = qubitsCombinations[bb]
    QubitsN = sum(ValoresSubQubits)
    profundidad = 3*QubitsN
    
    circuitos_r = []
    for pp in range(NumberOfStates):
        circuitos_r.append(QuantumCircuit(QubitsN))
        add_random_haar_gates(circuitos_r[pp])
        #circuitos_r.append(GenerateHaar(QubitsN))
    #circuitos_r = np.array(circuitos_r) 
    #dump(circuitos_r, "circuitos.mmap")

    #circuitos_shared = load("circuitos.mmap", mmap_mode ='r')


    #Lo que esta aqui abajo es para activar circuitos random de clifford
    #circuitos_r = [QuantumCircuit(QubitsN) for hh in range(NumberOfStates)]
    #for pp in range(NumberOfStates):
    #    for ll in range(profundidad):
    #        gate, n_qubits = random.choice(gates)
    #        puertas_qubits = random.sample(range(QubitsN), n_qubits)
    #        getattr(circuitos_r[pp], gate)(*puertas_qubits)

    for ss in range(len(shotsperWholeSystem)):

        
        #ValoresSubQubits = qubitsCombinations[bb]        ##EL 0 SE USA, NO QUITAR, DESPUES DEL CERO COLOCAR VALORES DE MAYOR A MENOR (EJ: [0,8,4,3,2])
        #NumberOfStates = 10
        tiros = shotsperWholeSystem[ss]
        #tiros = shotsperWholeSystem[bb]
        #QubitsN = sum(ValoresSubQubits)
        A=[]
        B=[]
        Bases = []
        BasesLimpias = []
        Empty = [0]*(len(ValoresSubQubits)-1)
        composedBases = []
        
        startingPoint = 0
        for ii in range(1,len(ValoresSubQubits)):    
            Alpha, Beta = AlgoritmoCircuitosMubs(ValoresSubQubits[ii])
            A.append(Alpha)
            B.append(Beta)
            Base = [CrearCircuito(A[ii-1],B[ii-1],ValoresSubQubits[ii], jjj,startingPoint) for jjj in range(2**ValoresSubQubits[ii]+1)]
            Limpiando = [CrearCircuito(A[ii-1],B[ii-1],ValoresSubQubits[ii], jjj,0, True) for jjj in range(2**ValoresSubQubits[ii]+1)]
            Bases.append(Base)
            BasesLimpias.append(Limpiando)
            startingPoint = startingPoint + ValoresSubQubits[ii]
        
                
        
            
        loop_rec(Bases,len(ValoresSubQubits)-1,Empty)
        #print(BasesLimpias)
        #print(len(composedBases))
        #composedBases[-1].draw('mpl')
        #str(Statevector(composedBases[2]))
        #Bases[2][2].draw('mpl')
        
        #print(len(composedBases))
        
        
        mubs = composedBases
        
        Fidelidades = []
        error2 = []
        errornuc = []
        forshots=1 #DONT TOUCH THIS VALUE
        for ww in range(1,len(ValoresSubQubits)):
            forshots = (2**ValoresSubQubits[ww]+1)*forshots
        
        
        negatividades = []
        theory_negativity = []
        #for qq in range(NumberOfStates):
            #profundidad = 3*QubitsN
            #tomo, qc, random_qc = GenerateRandomCircuit(profundidad, False)
            
            #sampler = Sampler()
            #job = sampler.run( tomo , shots=tiros/forshots )
            #probs_dicts = job.result().quasi_dists
            #print(len(probs_dicts))
            #print(BasesLimpias,probs_dicts)
            #if (len(ValoresSubQubits) == 2):
            #    densityMatrix = standardMUBTomography(probs_dicts, QubitsN, mubs)
            #else:
                #densityMatrix = TomographyMUBS(BasesLimpias,ValoresSubQubits,probs_dicts)
            #densityMatrix = DensityResonstructionMSystemsMubs(BasesLimpias,ValoresSubQubits,probs_dicts)


            
            #print(densityMatrix)
            #densityMatrix = ProyectForPositiveEigenValues(densityMatrix)
            #state = np.reshape(densityMatrix,(2**QubitsN,2**QubitsN))
            #densityMatrix = DensityMatrix(densityMatrix)
            #densityMatrixTheory_statevector = Statevector(random_qc)
            #densityMatrixTheory = DensityMatrix(densityMatrixTheory_statevector)

            
            #listForPartialTranspose =  list(range(ValoresSubQubits[1], sum(ValoresSubQubits)))


            #theory_negativity.append(NegativityOfDensityMatrix(densityMatrixTheory,listForPartialTranspose, ValoresSubQubits[-1]))
            #negatividades.append(NegativityOfDensityMatrix(densityMatrix,listForPartialTranspose, ValoresSubQubits[-1])) 
            #Fidelidades.append(state_fidelity(state, densityMatrixTheory_statevector, validate=False))
            #error2.append(np.linalg.norm(np.subtract(densityMatrixTheory,state),2)**2)
            #errornuc.append(np.linalg.norm(np.subtract(densityMatrixTheory, state), ord="nuc")**2)   #CUADRADO, SI VAS A ACAMBIARLO ACTUALIZA ESTA LINEA Y COMENTARIO
            
        #profundidad = 3*QubitsN
        #circuitos_r = [random_circuit(QubitsN, profundidad) for hh in range(NumberOfStates)]

        resultados_tomo = Parallel(n_jobs=cores,prefer="threads",max_nbytes=None)(delayed(ParallelStates)(qq) for qq in range(NumberOfStates))
        
        theory_negativity = [r[0] for r in resultados_tomo]
        negatividades = [r[1] for r in resultados_tomo]
        Fidelidades = [r[2] for r in resultados_tomo]
        error2 = [r[3] for r in resultados_tomo]
        errornuc = [r[4] for r in resultados_tomo]
    
    
    
            
        #den = qiskit.quantum_info.DensityMatrix(random_qc).to_operator()
        #print(TraceDistance(den, state))
        promedio = sum(Fidelidades)/len(Fidelidades)
        #print(Fidelidades)
        #print(promedio)
        #print(np.std(Fidelidades))
        #tomo[-1].draw('mpl')



        matrix_nucnorm[bb][ss] = sum(errornuc)/len(errornuc)
        matrix_nuc_std[bb][ss] = np.std(errornuc)
        matrix_infnorm[bb][ss] = sum(error2)/len(error2)
        matrix_inf_std[bb][ss] = np.std(error2)
    #stored_shots.append(tiros)

    
    
    stored_sistem_distribution.append(np.delete(ValoresSubQubits,0))
    stored_fidelities_mean.append(promedio)
    stored_std.append(np.std(Fidelidades))
    #print(np.subtract(densityMatrixTheory, state))
    #stored_2norm.append(sum(error2)/len(error2))
    #stored_nucnorm.append(sum(errornuc)/len(errornuc))
    stored_total_qubits.append(2**sum(ValoresSubQubits))
    stored_projected_negativity.append(sum(negatividades)/len(negatividades))
    stored_theory_negativity.append(sum(theory_negativity)/len(theory_negativity))
    stored_negativity.append(np.abs(sum(theory_negativity)/len(theory_negativity)-sum(negatividades)/len(negatividades)))##No tiene el cuadrado esta vez
    negatividades_para_std = []
    for nn in range(len(negatividades)):
        negatividades_para_std.append(np.abs(theory_negativity[nn]-negatividades[nn])) ##No tiene el cuadrado

    stored_negativity_std.append(np.std(negatividades_para_std))

matrix_nuc_std = matrix_nuc_std.T
matrix_inf_std = matrix_inf_std.T
stored_shots = shotsperWholeSystem
    
with open("distributed_tomography_data.txt", "w") as f:
    f.write(
                "stored_sistem_distribution = [" +
                    ", ".join(f"np.{repr(x)}" for x in stored_sistem_distribution) +
                        "]\n"
                        )
    #f.write(f"stored_sistem_distribution = {stored_sistem_distribution}\n")
    f.write(f"stored_fidelities_mean = {stored_fidelities_mean}\n")
    f.write(f"stored_std = {stored_std}\n")
    f.write(f"stored_total_qubits = {stored_total_qubits}\n")
    f.write(f"stored_projected_negativity = {stored_projected_negativity}\n")
    f.write(f"stored_theory_negativity = {stored_theory_negativity}\n")
    f.write(f"stored_negativity = {stored_negativity}\n")
    f.write(f"stored_shots = {stored_shots}\n")
    f.write(f"shotsperWholeSystem = stored_shots\n")
    f.write(f"matrix_infnorm = {matrix_infnorm.tolist()}\n")
    f.write(f"matrix_inf_std = {matrix_inf_std.tolist()}\n")
    f.write(f"matrix_nucnorm = {matrix_nucnorm.tolist()}\n")
    f.write(f"matrix_nuc_std = {matrix_nuc_std.tolist()}\n")
    f.write(f"stored_negativity_std = {stored_negativity_std}\n")
