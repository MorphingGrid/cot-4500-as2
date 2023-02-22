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
            second_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            # print(second_multiplication - first_multiplication)

            denominator = x_points[i] - x_points[i-j]
            #print(denominator)

            # this is the value that we will find in the matrix
            coefficient = (second_multiplication - first_multiplication)/denominator
            matrix[i][j] = coefficient
            
    return matrix

"""Q2) Using Newton's Forward Method, print out the polynomial approximation for degrees 1,2,
    and 3 using the following set of data"""
    
def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
    matrix: np.array = np.zeros([size,size])

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i+1):
            # the numerator are the immediate left and diagonal left indices...
            numerator =  matrix[i][j-1] - matrix[i-1][j-1]

            # the denominator is the X-SPAN...
            denominator = x_points[i]-x_points[i-j]

            operation = numerator / denominator

            # cut it off to view it more simpler
            matrix[i][j] = operation


    return matrix
    
"""Using Results from Question 2, Approximate f(7.3)"""
def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x 
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    
    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        
        polynomial_coefficient = matrix[index][index]
        #print("This is the coefficient: {}".format(polynomial_coefficient))

        # we use the previous index for x_points....
        reoccuring_x_span *= (value - x_points[index-1])
        
        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span
        #print("This is the operation: {}".format(mult_operation))

        # add the reoccuring px result
        reoccuring_px_result += mult_operation

    
    # # final result
    return reoccuring_px_result

"""Using the divided difference method, print out the Hermite polynomial approximation"""
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            # get left cell entry
            left: float = matrix[i][j-1]

            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]

            # order of numerator is SPECIFIC.
            numerator: float = left - diagonal_left

            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i-(j-1)][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix


def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]

    # matrix size changes because of "doubling" up info for hermite 
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points*2, num_of_points*2))

    x_points_double = np.repeat(x_points,2)
    for x in range(0,matrix.shape[0]):
        matrix[x][0] = x_points_double[x]
        
    y_points_double = np.repeat(y_points,2)
    for x in range(0,matrix.shape[0]):
        matrix[x][1] = y_points_double[x]
        
    for x in range(1, matrix.shape[0],2): 
        matrix[x,2] = slopes[x//2]
        

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix, end = '\n\n')

"""Using cubic spline interpolation, solve for the following using this set of data"""

def cubic_spline(x,y):
    #Find Matrix A
    size = len(x)
    matrix = np.zeros((size,size))
    matrix[0,0] = 1
    matrix[size-1,size-1] = 1
    
    for i in range(1, size-1):
        matrix[i, i-1] = x[i] - x[i-1]
        matrix[i, i] = 2 * (x[i+1] - x[i-1])
        matrix[i, i+1] = x[i+1] - x[i]
    
    print(matrix, end = '\n\n')
    
    #Find Vector B
    matrix1 = np.zeros(size)
    
    for i in range(1, size-1):
        matrix1[i] = 3 * (y[i+1] - y[i]) / (x[i+1] - x[i]) - \
            3 * (y[i] - y[i-1]) / (x[i] - x[i-1])
    
    print(matrix1, end = '\n\n')
    
    #Find Vector x
    matrix_x = np.linalg.solve(matrix,matrix1)
    
    print(matrix_x, end = '\n\n')
    
    
    

if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    
    ##Q1
    x_points = [3.6,3.8,3.9]
    y_points = [1.675,1.436,1.318]
    approximating_value = 3.7

    matrix = nevilles_method(x_points, y_points, approximating_value)
    print(matrix[2,2],end = '\n\n')
    
    ##Q2
    x_points = [7.2,7.4,7.5,7.6]
    y_points = [23.5492,25.3913,26.8224,27.4589]
    divided_table = divided_difference_table(x_points, y_points)
    ans = [divided_table[1,1],divided_table[2,2],divided_table[3,3]]
    print(ans, end = '\n\n')
    
    #Q3
    approx = 7.3
    answer = get_approximate_result(divided_table,x_points,approx)
    print(answer, end = '\n\n')
    
    #Q4
    hermite_interpolation()
    
    #Q5
    x = [2,5,8,10]
    y = [3,5,7,9]
    cubic_spline(x,y)
    
    
    
    