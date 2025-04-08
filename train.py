import numpy as np
from datasets import load_dataset

dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')

print(dataset['train']['features'].image)