# Проверка корректности ввода для задачи шифровщика
def shifr_is_valid():
    global a
    while True:
        a = input("Если нужно зашифровать текст - введи 1, если дешифровать - введи 0:")
        if a == "1" or a == "0":
            return True
        else:
            print("нужно ввести 0 или 1")
            continue


# Проверка корректности ввода языка текста
def leng_is_valid():
    global leng
    while True:
        leng = input('Если язык текста русский введи "ru", если английский введи "en":')
        if leng.lower() == "ru" or leng.lower() == "en":
            return True
        else:
            print('Нужно ввести "ru" или "en"')
            continue


# Функция шифрования/дешифрования английских строк
def en_shifr(shifr, string):
    unswer = ""
    alpha = [c for c in string if c.isalpha()]
    push = len(alpha)
    for c in string:
        if c.isalpha():
            if c.isupper():
                if shifr:
                    sim = ord(c) + push
                    if sim > 90:
                        sim -= 26
                    unswer += chr(sim)
                if not shifr:
                    sim = ord(c) - push
                    if sim < 65:
                        sim += 26
                    unswer += chr(sim)
            else:
                if shifr:
                    sim = ord(c) + push
                    if sim > 122:
                        sim -= 26
                    unswer += chr(sim)
                if not shifr:
                    sim = ord(c) - push
                    if sim < 97:
                        sim += 26
                    unswer += chr(sim)
        else:
            unswer += c
    return unswer


# Функция шифрования/дешифрования русских строк
def ru_shifr(shifr, string):
    unswer = ""
    alpha = [c for c in string if c.isalpha()]
    push = len(alpha)
    for c in string:
        if c.isalpha():
            if c.isupper():
                if shifr:
                    sim = ord(c) + push
                    if sim > 1071:
                        sim -= 32
                    unswer += chr(sim)
                if not shifr:
                    sim = ord(c) - push
                    if sim < 1040:
                        sim += 32
                    unswer += chr(sim)
            else:
                if shifr:
                    sim = ord(c) + push
                    if sim > 1103:
                        sim -= 32
                    unswer += chr(sim)
                if not shifr:
                    sim = ord(c) - push
                    if sim < 1072:
                        sim += 32
                    unswer += chr(sim)
        else:
            unswer += c
    return unswer


# Основная функция программы
def main():
    global shifr, string
    if shifr_is_valid() and a == "1":
        shifr = True
    else:
        shifr = False
    leng_is_valid()
    string = input("Введите строку для шифровки:")
    get_list = string.split(" ")
    ending = ""
    if leng == "ru":
        for c in get_list:
            ending += ru_shifr(shifr, c)
            ending += " "
        ending.rstrip()
        print(ending)
    elif leng == "en":
        for c in get_list:
            ending += en_shifr(shifr, c)
            ending += " "
        ending.rstrip()
        print(ending)


if __name__ == "__main__":
    main()
