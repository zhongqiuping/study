import string
import math
a = '12345'
a = a[2:4]
b = len(a)
print(b)
print(a)
print("{1}:计算机{2}的CPU的{0}".format("python", "你好", 10))
# 数字输出（5位数的个位，十位……）
def outputnumber():

    number = input("请输入5位数:")
    strnumber = str(number)
    if len(strnumber)!=5:
        print("输入位数不正确请重新输入")
    else:print("个位数{}，十位数{}，百位数{}，千位数{}，万位数{}".format(strnumber[4], strnumber[3], strnumber[2], strnumber[1], strnumber[0]))
# BMI计算
def BMIcount():

    weight = input("请输入体重(KG)：")
    heigh = input("请输入身高(M):")
    Nmheight, Nmweight = float(heigh), float(weight)
    BMI=float(Nmweight/(Nmheight*Nmheight))
    print("您的BMI指数位{}".format(BMI))
    if BMI < 18.5:
        CN = "偏瘦"
    elif BMI < 24:
        CN = "健康"
    elif BMI < 30:
        CN = "超重"
    else:
        CN = "肥胖"
    print("您的BMI指数为：{}，肥胖程度:{}".format(BMI, CN))


# 统计长度(统计任意数的20次幂的值和位数
def Countlen():
    dishu = input("输入底数：")
    dishucifang = int(dishu)**20
    strdishucifang = str(dishucifang)
    lencifang = len(strdishucifang)
    print("底数{}的20次幂为：{}，该数的位数位：{}".format(dishu, dishucifang,lencifang))
# 加密
def password():
    IntPassword = input("请输入四位数密码明文：")
    # 元组
    x1, x2, x3, x4 = IntPassword[0], IntPassword[1], IntPassword[2], IntPassword[3]
    z1, z2, z3, z4=int(x1), int(x2), int(x3), int(x4)
    y1, y2, y3, y4=(z1+5) % 10, (z2+5) % 10, (z3+5) % 10, (z4+5) % 10
    print("输出密码：{},{},{},{}".format(y1,y2,y3,y4))
if __name__ == '__main__':
    while(1):
        choose = input("输入你要选择的功能：1：数字输出;2：BMI计算;3：统计长度;4:加密")
        choose = int(choose)
        if choose == 1:
            outputnumber()
        elif choose == 2:
            BMIcount()
        elif choose == 3:
            Countlen()
        elif choose == 4:
            password()
        else:print("指令错误，请重新输入！")

