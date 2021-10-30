import os, glob
import os.path

def a():
    file_dir = "./data/stock"
    list_files = os.listdir(file_dir)
    print('\n')
    print('*'*10+'데이터 파일 목록'+'*'*10)

    a = []
    b = []
    file_list = {}
    for v in range(len(list_files)):
        a.append(v)
    a.append(len(a))

    for  i in  list_files :
        b.append(i)
    b.append("뒤로")

    for x in range(len(list_files)+1):
        file_list[a[x]] = b[x]

    for y in range(len(file_list)) :
        print(y ,":", file_list[y])

    print('*'*35)

    return file_list

b = a()
print(b[1])