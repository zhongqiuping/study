import turtle
from datetime import datetime
now = datetime.now()
print(now)
print(now.strftime('%x'))
turtle.goto(100, 100)
turtle.goto(100, -100)
turtle.pensize(2)
turtle.circle(20)
turtle.circle(40)
turtle.circle(80)
radius = 25
area = 3.1415*radius*radius
print(area)
print(".2f".format(area))
name = input("请输入姓名：")
print(name)
print("{}同学学习数学，前途无量！".format(name))
print(name+"同学学习数学，前途无量！")
print("{}同学学习数学，前途无量！".format(name[0]))
a = 0
b = 1
while a <1000:
    print(a, end=',')
    a, b = b, a+b

