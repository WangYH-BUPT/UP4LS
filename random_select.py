import random


name = "Selenagomez"  # 10
with open('./Dataset/twitter_20_top/' + name + '/unbalance_c/stego.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

random_lines = random.sample(lines, 10)
with open('./Dataset/twitter_20_top/' + name + '/unbalance_c/10stego.txt', 'w', encoding='utf-8') as file:
    file.writelines(random_lines)


random_lines = random.sample(lines, 20)
with open('./Dataset/twitter_20_top/' + name + '/unbalance_c/20stego.txt', 'w', encoding='utf-8') as file:
    file.writelines(random_lines)


random_lines = random.sample(lines, 30)
with open('./Dataset/twitter_20_top/' + name + '/unbalance_c/30stego.txt', 'w', encoding='utf-8') as file:
    file.writelines(random_lines)


print("save output.txt")
