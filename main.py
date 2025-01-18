import colour
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import ImageDraw
from PIL import Image

POPULATION_SIZE = 100
NUM_GENERATIONS = 2001
MUTATION_RATE = 0.1
LEN = 64
WID = 64
IMAGE = Image.open("original.png")


def create_population():
    population = [create_random_image() for _ in range(POPULATION_SIZE)]
    return population


def rand_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])


def create_random_image():
    """
    cria uma imagem com uma cor de fundo aleatória e adiciona alguns polígonos coloridos nela
    :return: PIL Image
    """
    iterations = random.randint(3, 6)

    region = (LEN + WID) // 8

    img = Image.new("RGBA", (LEN, WID), rand_color())

    for i in range(iterations):
        num_points = random.randint(3, 6)

        region_x = random.randint(0, LEN)
        region_y = random.randint(0, WID)

        xy = []
        for j in range(num_points):
            xy.append((random.randint(region_x - region, region_x + region),
                       random.randint(region_y - region, region_y + region)))

        img1 = ImageDraw.Draw(img)
        img1.polygon(xy, fill=rand_color())

    return img


def get_fitness(individual):
    """Calcula a fitness baseado na diferença entre as cores de cada píxel da imagem.
    Calculo é feito usando Delta E, que traz uma precisão maior do que só o valor rgb puro
    Já que o delta E traz um valor de diferença baseado na percepção do olho humano e não dos valores da cor
    Ex: laranja e marrom são duas cores muito próximas em questão de valores mas muito diferentes ao olhar"""
    fitness = np.mean(colour.difference.delta_e.delta_E_CIE1976(IMAGE, np.array(individual)))
    return fitness


#Seleção por torneio
def select_parents(population):
    indices = np.random.choice(len(population), 6)
    random_subset = [population[i] for i in indices]

    winner = None

    for i in random_subset:
        if winner is None:
            winner = i
        elif get_fitness(i) < get_fitness(winner):
            winner = i

    return winner

def crossover(parent1, parent2):
    """Faz um crossover pegando um número alpha aleatório e fundindo as duas imagens
    O alpha é aplicado como transparência das duas imagens"""
    alpha = random.random()
    child = mutate(Image.blend(parent1, parent2, alpha))
    fitness = get_fitness(child)
    if fitness == min(get_fitness(parent1), get_fitness(parent2), fitness):
        return child
    return None

def crossover_2(parent1, parent2, horizontal_prob):
    """

    :param parent1: pai 1
    :param parent2: pai 2
    :param horizontal_prob: probabilidade da imagem ser cortada horizontalmente (ex: 0.5 == 50% de chance
    da imagem ser cortada horizontalmente, 50% de chance dela ser cortada verticalmente
    :return: PIL Image da mistura dos dois pais
    """
    rand = random.random()

    #os cortes criam um novo array marcado com 1s e 0s, sendo a parte que estiver marcada com 1s a parte que será usada
    #é uma mascara para saber qual parte da imagem vai ser usada
    #corte horizontal
    if rand <= horizontal_prob:

        split_point = random.randint(1, WID)

        first = np.vstack((
            np.ones((split_point, LEN)),
            np.zeros((WID - split_point, LEN))
        ))

    #corte vertical
    else:
        split_point = random.randint(1, LEN)

        first = np.hstack((
            np.ones((WID, split_point)),
            np.zeros((WID, LEN - split_point))
        ))

    #indica qual parte do outro pai vai ser usada, sendo todas as partes não usadas do pai 1
    #ou seja, o inverso
    second = 1 - first

    #essa parte é que faz o array funcionar em todos os canais de cores (RGBA)
    first = np.dstack([first, first, first, first])
    second = np.dstack([second, second, second, second])

    #mantém os pixels onde é 1 no array
    half_chromo_1 = np.multiply(first, np.array(parent1))
    half_chromo_2 = np.multiply(second, np.array(parent2))

    child_array = np.add(half_chromo_1, half_chromo_2)

    child = mutate(Image.fromarray(child_array.astype(np.uint8)))

    fitness = get_fitness(child)
    #joga fora o filho se ele não for mais apto que o pai. acaba convergindo a maior parte dos filhos
    #mas sem isso a melhoria entre gerações fica nula ou pode até reverter
    #taxa de mutação alta cuida do problema da convergência
    if fitness == min(get_fitness(parent1), get_fitness(parent2), fitness):
        return child

    return None


def mutate(img):
    chance = random.random()
    if chance > MUTATION_RATE:
        return img
    iterations = random.randint(1, 1)
    region = random.randint(1, (LEN + WID) // 4)
    for i in range(iterations):
        num_points = random.randint(3, 6)
        region_x = random.randint(0, LEN)
        region_y = random.randint(0, WID)

        xy = []
        for j in range(num_points):
            xy.append((random.randint(region_x - region, region_x + region),
                       random.randint(region_y - region, region_y + region)))

        img1 = ImageDraw.Draw(img)
        img1.polygon(xy, fill=rand_color())
    return img


def genetic_algorithm():
    #gerar população
    population = create_population()

    for i in range(NUM_GENERATIONS):
        new_population = []

        while len(new_population) < len(population):
            rand = random.random()

            #seleciona qual método de crossover será utilizado
            if rand < 0.35:
                child = None

                while child is None:
                    parent_one = select_parents(population)
                    parent_two = select_parents(population)

                    child = crossover(parent_one, parent_two)

            else:
                child = None

                while child is None:
                    parent_one = select_parents(population)
                    parent_two = select_parents(population)

                    child = crossover_2(parent_one, parent_two, 0.5)


            new_population.append(child)

        population = new_population

        #salva imagem a cada 100 gerações ou se for a última
        if i % 50 == 0 or i == NUM_GENERATIONS - 1:
            population.sort(key=lambda ind: get_fitness(ind))
            fittest = population[0]
            print("Most fit individual in epoch " + str(i) +
                " has fitness: " + str(get_fitness(fittest)))

            fittest.save("generatedImages/fittest_" + str(i) + ".png")

    population.sort(key=lambda ind: get_fitness(ind))
    fittest = population[0]
    return fittest

if __name__ == "__main__":
    reconstructed_image = genetic_algorithm()
    plt.imshow(reconstructed_image)
    plt.title("Imagem Reconstruída")
    plt.savefig('reconstructed_image.png')