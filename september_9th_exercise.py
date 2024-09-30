# Fibonnacci sequence

def Fibonnacci(): 
    a,b = 0,1
    n = eval(input("Please enter an integer: "))
             
    while a<n:
        print(a)
        a,b = b, a+b

Fibonnacci()

    