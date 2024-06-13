import timeit

# Original list
original_list = list(range(1000000))  # Adjust the size for more significant results

# Method 1: Using the copy() method
def copy_method():
    copied_list = original_list.copy()

# Method 2: Using slicing
def slice_method():
    copied_list = original_list[:]

# Method 3: Using the list constructor
def constructor_method():
    copied_list = list(original_list)

# Method 4: Using the copy module
import copy
def copy_module_method():
    copied_list = copy.copy(original_list)

# Run timeit for each method
iterations = 1000
time_copy = timeit.timeit(copy_method, number=iterations)
time_slice = timeit.timeit(slice_method, number=iterations)
time_constructor = timeit.timeit(constructor_method, number=iterations)
time_copy_module = timeit.timeit(copy_module_method, number=iterations)

print(f'copy() method: {time_copy:.6f} seconds')
print(f'Slicing: {time_slice:.6f} seconds')
print(f'list() constructor: {time_constructor:.6f} seconds')
print(f'copy module: {time_copy_module:.6f} seconds')
