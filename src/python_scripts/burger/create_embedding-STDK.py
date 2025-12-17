
#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday Jan  30 2025

@author: Pratik
"""

import numpy as np

def main():
    time_points = 5
    space_points = 2**11
    time_pts_test = 1.0
    x = np.linspace(0, 1, space_points)    # shape: (2048,)
    t = np.linspace(0, 1, time_points)     # shape: (time_points,)

    X, T = np.meshgrid(x, t, indexing='ij')
    coords = np.stack([X.ravel(), T.ravel()], axis=-1)  # shape: (20480, 2)
    ntrain = len(coords)
    t_test = np.array([time_pts_test])
    X_test, T_test = np.meshgrid(x, t_test, indexing='ij')  # shape: (2048, 1)
    coords_test = np.stack([X_test.ravel(), T_test.ravel()], axis=-1)  # shape: (2048, 2)
    all_coords = np.vstack([coords, coords_test])
    # Space basis
    num_basis_space = [7**2, 11**2, 17**2]
    knots_1d_space = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis_space]
    
    # Prepare the output CSV file
    # List to collect the rows for phi_t and phi
    data_list = []
    
    # For each row in the dataset, calculate the phi_t and phi and store them
    for j in range(len(all_coords)):
    
        # Calculate space basis (phi)
        phi_row = np.zeros((1, sum(num_basis_space)))
        s_j_2d = all_coords[j,:]
        K_space = 0
        for res in range(len(num_basis_space)):
            theta = 1 / np.sqrt(num_basis_space[res]) * 2.5
            knots_s1, knots_s2 = np.meshgrid(knots_1d_space[res], knots_1d_space[res])
            knots = np.column_stack((knots_s1.flatten(), knots_s2.flatten()))
            
            for i in range(num_basis_space[res]):
                d = np.linalg.norm(s_j_2d - knots[i, :]) / theta
                if d >= 0 and d <= 1:
                    phi_row[0,i+K_space]=((1 - d) ** 6 * (35 * d ** 2 + 18 * d + 3) / 3)
                else:
                    phi_row[0,i+K_space] = 0
            K_space += num_basis_space[res]
    
        # Convert the filtered row to float16
        filtered_row_float16 = np.array(phi_row, dtype=np.float16)
    
        # Append the row to the data list
        data_list.append(filtered_row_float16)
        if(j % 1000 == 0):
            print('{}-th row done'.format(j))
    # Convert the list of rows into a NumPy array
    data_array = np.vstack(data_list)
    del(data_list)
    print(data_array.shape)
    ## Romove the all-zero columns
    idx_zero = np.array([], dtype=int)
    for i in range(data_array.shape[1]):
        if sum(data_array[:,i]!=0)==0:
            idx_zero = np.append(idx_zero,int(i))
    
    phi_reduce = np.delete(data_array,idx_zero,1)
    print(phi_reduce.shape)
    phi_train = phi_reduce[:ntrain,:]
    phi_test = phi_reduce[ntrain:,:]

    # Save the resulting array as a .npy file
    output_file = 'datasets/phi_float16_train-burger.npy'
    np.save(output_file, phi_train)

    output_file = 'datasets/phi_float16_test-burger.npy'
    np.save(output_file, phi_test)

    output_file = 'datasets/index_zero-burger.npy'
    np.save(output_file, idx_zero)
    
    print(f"Data has been written to {output_file}")

if __name__ == '__main__':
    main()
