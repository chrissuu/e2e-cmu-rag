import random
random.seed(19)

f = open("collated_questions.txt", "r")
iaa = open("iaa.txt", "w")
lines = f.readlines()
randomly_sampled = random.sample(lines, int(0.3 * len(lines)))
for line in randomly_sampled:
    iaa.write(line)
