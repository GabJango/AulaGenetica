from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.analysis.fitness_store import FitnessStore

# bibliotecas
from math import cos, sin
import matplotlib.pyplot as plt
import numpy as np

# Parametros para geração dos individuos
individuo_template = BinaryIndividual(ranges=[(0, 10)], eps=0.001)

# Parametros para inicializar a população
population = Population(indv_template=individuo_template, size=50)

# Parametros para inicializar a população
population.init()

# Parametros para execução dos operadores
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population=population,
                  selection=selection, crossover=crossover, mutation=mutation, analysis=[FitnessStore])


# função de aptidão
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return x + 10 * sin(5 * x) + 7 * cos(4 * x)


@engine.analysis_register
class ConsoleOutPut(OnTheFlyAnalysis):
    interval = 1

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Geração: {}, best fitness {:.3f}'.format(g, engine.fmax)
        engine.logger.info(msg)


def finalize(population, engine):
    best_indv = population.best_indv(engine.fitness)
    x = best_indv.solution
    y = engine.ori_fmax
    msg = 'Solução ótima: ({},{})'.format(x, y)
    engine.logger.info(msg)


def plotAnalise():
    geracao = []  # variavel contendo a lista de gerações
    fitness = []  # variavel contendo a lista de fitness

    #Lista2
    geracao2 = []  # variavel contendo a lista de gerações
    fitness2 = []  # variavel contendo a lista de fitness

    best_fit_file = open("best_fit1.py", "r")

    #Lista2
    best_fit_file2 = open("best_fit2.py", "r")

    for linha in best_fit_file:
        novalinha = linha.replace('[', "")
        novalinha = novalinha.replace('(', "")
        novalinha = novalinha.replace(')', "")
        novalinha = novalinha.replace(',', "")
        novalinha = novalinha.replace(']', "")
        lista = novalinha.split()
        geracao.append(int(lista[0]))  # prencher a lista de geracoes
        fitness.append(float(lista[2]))  # prencher a lista de fitness
    #Lista2
    for linha2 in best_fit_file2:
        novalinha2 = linha2.replace('[', "")
        novalinha2 = novalinha2.replace('(', "")
        novalinha2 = novalinha2.replace(')', "")
        novalinha2 = novalinha2.replace(',', "")
        novalinha2 = novalinha2.replace(']', "")
        lista2 = novalinha2.split()
        geracao2.append(int(lista2[0]))  # prencher a lista2 de geracoes2
        fitness2.append(float(lista2[2]))  # prencher a lista2 de fitness2

    #Primeiro Gráfico
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(geracao, fitness)
    ax.set_xlabel('Geraçoes')
    ax.set_ylabel('Fitness')
    plt.title("Grafico de Gerações por Fitness")


    #Segundo Gráfico
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(geracao2, fitness2)
    ax.set_xlabel('Geraçoes2')
    ax.set_ylabel('Fitness2')
    plt.title("Grafico2 de Gerações2 por Fitness2")

    #Mostrar Gráficos
    plt.show()


## if '__main__' == __name__:
## engine.run(ng=100)

plotAnalise()