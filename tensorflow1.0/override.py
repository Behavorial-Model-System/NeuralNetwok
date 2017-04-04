class Box:
    height = 0

    def show(self, num):
        print('a')

box1 = Box()
def newShow(num):
    print(num)
box1.show = newShow
box1.show(1)


box2 = Box()
def newShow2(num):
    print(num*2)
box2.show = newShow2


box2.show(1)

box1.show(1)