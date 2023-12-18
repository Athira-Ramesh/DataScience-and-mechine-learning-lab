import numpy as np

def inputMatrix(matrix_name, row, col):
    mat = []
    print(f"enter the elements for the {matrix_name}")
    for i in range(row):
        mat1 = []
        for j in range(col):
            n = int(input(f"enter the row {i + 1} and column {j + 1}"))
            mat1.append(n)
        mat.append(mat1)
    return np.array(mat)

row = int(input("enter the number of rows"))
col = int(input("enter the number of columns"))
matrix1 = inputMatrix("matrix1", row, col)
matrix2 = inputMatrix("matrix2", row, col)

print(matrix1)
print(matrix2)

print("matrix multiplication\n", np.multiply(matrix1, matrix2))
print("matrix subtraction\n", np.subtract(matrix1, matrix2))
print("matrix division\n", np.divide(matrix1, matrix2))
print("matrix addition\n", np.add(matrix1, matrix2))
print("dot product matrix1\n", np.dot(matrix1, matrix2))
print("transpose", np.transpose(matrix1))
print("matrix squareroot\n", np.sqrt(matrix1))
print("matrix square\n",np.square(matrix1))