import random, sys

secret = random.randint(1,100)
count = 0
print('猜1-100之间的整数，q推出')

while True:
    guess = input('>> ').strip()
    if guess.lower() == 'q':
        sys.exit('拜拜~~~~~~~')

    if not guess.isdigit():
        print('请输入数字')
        continue

    guess = int(guess)
    count += 1

    if guess > secret:
        print('----------------\n|      大了      |\n----------------')

    elif guess < secret:
        print('----------------\n|      小了      |\n----------------')

    else:
        print(f'----------------------\n|     猜对啦！共用了 {count} 次      |\n--------------------------')
        break    
