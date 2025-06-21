import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data_Purify():
    def __init__(self,filename,xr,p_max,u_max) -> None:

        # Load the CSV dataset
        df = pd.read_csv(filename, header=None)

        # Format:
        # x1 p1 p2 u
        # x2 p2 p3 theta
        self.purify_factor_p = p_max
        self.purify_factor_u = u_max

        # Assuming the first row is the header, adjust if needed
        data = df.values  # Transpose to have shape (2, n)


        self.data_x = []
        self.data_u = []
        self.data_p = []
        self.data_theta = []
        instance = 2
        for row in range(data.shape[0]//instance):
            self.data_x.extend([data[instance*row:instance*row+instance,0]])
            self.data_u.extend([data[instance*row,3]])
            self.data_p.extend([data[instance*row:instance*row+instance,1:3]])
            self.data_theta.extend([data[instance*row+1,3]])

        
        # self.data_theta = np.tile(xr,(len(self.data_u),1))
        print('----------DATA SUMMARY------------')
        print(f'There are {len(self.data_u)} raw data points')

        # combine [x1,x2] and r as a single input:
        # self.input_data_combine = [np.append(self.data_x[i],self.data_theta[i]) for i in range(len(self.data_x))]
        self.input_data_combine = [np.append(self.data_x[i],0) for i in range(len(self.data_x))]

        # now we only need 4: p1,p2,p3,..., p10, u, theta
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],self.data_p[i][1,1],
                            self.data_u[i], self.data_theta[i]] for i in range(len(self.data_u))]

    def return_data(self,):
            return self.input_data_combine, self.label_data



    def purified_data(self):

        data_p1 = [x[0,0] for x in self.data_p]
        data_p2 = [x[0,1] for x in self.data_p]
        data_p3 = [x[1,1] for x in self.data_p]

        data_p1 = np.array(data_p1)
        data_p2 = np.array(data_p2)
        data_p3 = np.array(data_p3)

        self.data_u = np.array(self.data_u)
        mask_outlier1 = np.abs(data_p1) < (self.purify_factor_p)
        mask_outlier3 = np.abs(data_p2) < (self.purify_factor_p)
        mask_outlier4 = np.abs(data_p3) < (self.purify_factor_p)
        mask_outlier2 = np.abs(self.data_u) < (self.purify_factor_u)
        self.mask_outlier = np.logical_and(
            np.logical_and(mask_outlier1,mask_outlier2),
            np.logical_and(mask_outlier3,mask_outlier4)
        )

        # for outliers
        self.outlier_mask = np.logical_not(self.mask_outlier)
        self.outliers = np.array(self.data_x)[self.outlier_mask]
        # use mask get rid of outliers
        self.data_p = np.array(self.data_p)[self.mask_outlier]
        self.data_theta = np.array(self.data_theta)[self.mask_outlier]
        self.data_x = np.array(self.data_x)[self.mask_outlier]
        self.data_u = np.array(self.data_u)[self.mask_outlier]
        print(f'Dataset is purified! Now there are {len(self.data_theta)} data points available.')
        print('--------------------------------')

        self.len_data = len(self.data_u)

        # Final Data for feeding NN.

        # combine [x1,x2] and r as a single input:
        self.input_data_combine = [np.append(self.data_x[i],0) for i in range(len(self.data_x))]

        # now we only need 5: p1,p2,p3,u, theta
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],
                            self.data_p[i][1,1],self.data_u[i],self.data_theta[i]] for i in range(len(self.data_u))]

        return self.input_data_combine, self.label_data


    def draw_data(self, data_u, data_p):

        plt.subplot(221)
        plt.plot(list(range(len(data_u))),data_u)
        # plt.show()
        plt.title('Parameter u(t)')
        # plt.close()

        plt.subplot(222)
        data_p1 = [x[0,0] for x in data_p]
        plt.plot(list(range(len(data_p1))),data_p1)
        # plt.show()
        plt.title('Parameter p(1)')

        plt.subplot(223)
        data_p2 = [x[0,1] for x in data_p]
        plt.plot(list(range(len(data_p2))),data_p2)
        # plt.show()
        plt.title('Parameter p2')

        plt.subplot(224)
        data_p3 = [x[1,1] for x in data_p]
        plt.plot(list(range(len(data_p3))),data_p3)
        plt.show()
        plt.title('Parameter p3')
        plt.close()
