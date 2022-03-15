import math
import random
radius = float(input("请输入园的半径："))
circumference = 2*math.pi*radius
area = math.pi*radius*radius
print("圆的面积为:", area)
print("圆的周长为:", circumference)
print("随机1个[0,1)的小数：{:.1f}".format(random.random()))
radiu1 = random.uniform(0, 10)
high = random.uniform(0, 10)
V = math.pi*radiu1*radiu1*high/3
print("随机数的圆柱体体积：", V)
def jiecheng():
    n = int(input("输入阶乘n："))
    t = 1
    for i in range(1, n+1):
        t = t*i
    print("阶乘结果为：", t)

def shuixianhua():
    hua = input("输入一个三位数：")
    count = 0
    for i in hua:count = int(i)*int(i)+count
    print("水仙花各位数平方和：", count)

def sushu():
    N = int(input("判断下列数字是否为素数："))
    for i in(2, N-1):
        t = N % i
        if t == 0:
            print("该数为非素数")
            break
        else:
            print("该数为素数")

def chuangguan():
    counttrue = 0
    for i in range(0,5):
        x1 = random.randint(0, 100)
        x2 = random.randint(0, 100)
        sum = x1 + x2
        print("请您计算{}+{}=？".format(x1, x2))
        summan = int(input())
        if summan == sum: counttrue = counttrue+1
        print("正确结果为：", sum)
    print("您的正确次数为；", counttrue)
    if counttrue > 2:
        print("恭喜你闯关成功")
    else:
        print("还要继续努力啊")


if __name__ == '__main__':
    while(1):
        choose = input("输入你要选择的功能：1:阶乘计算;2：水仙花各位数和;3：判断数字素否;4:闯关游戏")
        choose = int(choose)
        if choose == 1:
            jiecheng()
        if choose == 2:
            shuixianhua()
        if choose == 3:
            sushu()
        if choose == 4:
            chuangguan()