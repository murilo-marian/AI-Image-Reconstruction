import colour
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import ImageDraw
from PIL import Image

def create_population(pop_size, width, height):
    population = [create_random_image(width, height) for _ in range(pop_size)]
    return population


def rand_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])


def create_random_image(width, height):
    """
    cria uma imagem com uma cor de fundo aleatória e adiciona alguns polígonos coloridos nela
    :return: PIL Image
    """
    iterations = random.randint(3, 6)

    region = (width + height) // 8

    img = Image.new("RGBA", (width, height), rand_color())

    for i in range(iterations):
        num_points = random.randint(3, 6)

        region_x = random.randint(0, width)
        region_y = random.randint(0, height)

        xy = []
        for j in range(num_points):
            xy.append((random.randint(region_x - region, region_x + region),
                       random.randint(region_y - region, region_y + region)))

        img1 = ImageDraw.Draw(img)
        img1.polygon(xy, fill=rand_color())

    return img


def get_fitness(individual, image):
    """Calcula a fitness baseado na diferença entre as cores de cada píxel da imagem.
    Calculo é feito usando Delta E, que traz uma precisão maior do que só o valor rgb puro
    Já que o delta E traz um valor de diferença baseado na percepção do olho humano e não dos valores da cor
    Ex: laranja e marrom são duas cores muito próximas em questão de valores mas muito diferentes ao olhar"""
    fitness = np.mean(colour.difference.delta_e.delta_E_CIE1976(image, np.array(individual)))
    return fitness


#Seleção por torneio
def select_parents(population, image):
    indices = np.random.choice(len(population), 6)
    random_subset = [population[i] for i in indices]

    winner = None

    for i in random_subset:
        if winner is None:
            winner = i
        elif get_fitness(i, image) < get_fitness(winner, image):
            winner = i

    return winner

def crossover(parent1, parent2, image, mutation_rate, width, height):
    """Faz um crossover pegando um número alpha aleatório e fundindo as duas imagens
    O alpha é aplicado como transparência das duas imagens"""
    alpha = random.random()
    child = mutate(Image.blend(parent1, parent2, alpha), mutation_rate, width, height)
    fitness = get_fitness(child, image)
    if fitness == min(get_fitness(parent1, image), get_fitness(parent2, image), fitness):
        return child
    return None

def crossover_2(parent1, parent2, horizontal_prob, image, mutation_rate, width, height):
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
    if rand <= horizontal_prob:

        split_point = random.randint(1, width)

        first = np.vstack((
            np.ones((split_point, height)),
            np.zeros((width - split_point, height))
        ))

    else:
        split_point = random.randint(1, height)

        first = np.hstack((
            np.ones((width, split_point)),
            np.zeros((width, height - split_point))
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

    child = mutate(Image.fromarray(child_array.astype(np.uint8)), mutation_rate, width, height)

    fitness = get_fitness(child, image)
    #joga fora o filho se ele não for mais apto que o pai. acaba convergindo a maior parte dos filhos
    #mas sem isso a melhoria entre gerações fica nula ou pode até reverter
    #taxa de mutação alta cuida do problema da convergência
    if fitness == min(get_fitness(parent1, image), get_fitness(parent2, image), fitness):
        return child

    return None


def mutate(img, mutation_rate, width, height):
    chance = random.random()
    if chance > mutation_rate:
        return img
    iterations = random.randint(1, 1)
    region = random.randint(1, (width + height) // 4)
    for i in range(iterations):
        num_points = random.randint(3, 6)
        region_x = random.randint(0, width)
        region_y = random.randint(0, height)

        xy = []
        for j in range(num_points):
            xy.append((random.randint(region_x - region, region_x + region),
                       random.randint(region_y - region, region_y + region)))

        img1 = ImageDraw.Draw(img)
        img1.polygon(xy, fill=rand_color())
    return img

def genetic_algorithm(pop_size, num_generations, mutation_rate, image):
    if image.mode == 'RGB':
        image = image.convert('RGBA')
    
    width, height = image.size
    
    #gerar população
    population = create_population(pop_size, width, height)

    for i in range(num_generations):
        new_population = []

        while len(new_population) < len(population):
            rand = random.random()

            #seleciona qual método de crossover será utilizado
            if rand < 0.35:
                child = None

                while child is None:
                    parent_one = select_parents(population, image)
                    parent_two = select_parents(population, image)

                    child = crossover(parent_one, parent_two, image, mutation_rate, width, height)

            else:
                child = None

                while child is None:
                    parent_one = select_parents(population, image)
                    parent_two = select_parents(population, image)

                    child = crossover_2(parent_one, parent_two, 0.5, image, mutation_rate, width, height)


            new_population.append(child)

        population = new_population

        #salva imagem a cada 100 gerações ou se for a última
        if i % 50 == 0 or i == num_generations - 1:
            population.sort(key=lambda ind: get_fitness(ind, image))
            fittest = population[0]
            """ print("Most fit individual in epoch " + str(i) +
                " has fitness: " + str(get_fitness(fittest, image))) """

            """ fittest.save("generatedImages/fittest_" + str(i) + ".png") """

    population.sort(key=lambda ind: get_fitness(ind, image))
    fittest = population[0]
    return fittest

if __name__ == "__main__":
    pop_size = 100
    num_generations = 100
    mutation_rate = 0.1
    image = Image.open("original.png")
    reconstructed_image = genetic_algorithm(pop_size, num_generations, mutation_rate, image)
    plt.imshow(reconstructed_image)
    plt.title("Imagem Reconstruída")
    plt.savefig('reconstructed_image.png')