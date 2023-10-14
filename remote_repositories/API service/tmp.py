import numpy as np

doors  =[1, 2, 3]

# функция выбора одной из трех дверей с равной вероятностью 
def select_door():
    return np.random.choice(doors)


