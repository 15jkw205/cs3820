'''
# Exercise 1 
def make_incrementor(n):
    return lambda x:x+n

f = make_incrementor(42)
print(f(0))
print(f(1))


# Exercise 2 
pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair:pair[1])
print(pairs)


# Exercise 3
def my_function():
    """Do nothing, but document it.

No really, it doesn't do anything.
    """
    
    pass

print(my_function.__doc__)
'''

# Group Exercise #1 

combined_list = []
list_one = [1,2,3]
list_two = [3,1,4]

for i in list_one:
    for j in list_two:
        
        if i != j:
            combined_list.append((i, j))

print(combined_list)

# Group Exercise #2
comp_list = [(i, j) for i in list_one for j in list_two if i != j]
other_list = list(map(lambda x:(i, j), for i in list_one for j in list_two if i != j))
            
            
