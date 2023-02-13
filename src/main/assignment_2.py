import numpy as np

"""Q1) Using Neville's Method, Find the 2nd degree interpolating value for f(3.7) for
        the following dataset"""
        
def nevilles_method(x_points, y_points, x):
    # must specify the matrix size (this is based on how many columns/rows you want)
    matrix = np.zeros((3, 3))

    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]

    # the end of the first loop are how many columns you have...
    num_of_points = len(x_points)

    # populate final matrix (this is the iterative version of the recursion explained in class)
    # the end of the second loop is based on the first loop...
    for i in range(1, num_of_points):
        for j in range(1, i+1):
            first_multiplication = (x - x_points[i]) * matrix[i-1][j-1]
            second_multiplication = (x - x_points[i-1]) * matrix[i][j-1]

            denominator = x_points[i] - x_points[i-1]

            # this is the value that we will find in the matrix
            coefficient = (second_multiplication - first_multiplication)/denominator
            matrix[i][j] = coefficient
            
    return matrix
    
if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    # point setup
    x_points = [3.6,3.8,3.9]
    y_points = [1.675,1.436,1.318]
    approximating_value = 3.7

    matrix = nevilles_method(x_points, y_points, approximating_value)
    print(matrix[2,2],sep = '\n\n')