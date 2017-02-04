"""
Intuit Challenge - Building Relationships
Author: Pavan Bhat (pxb8715.rit.edu)
"""

# All imports here
import os
import csv
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as npy


class clean_and_prep:
    '''
    This class is used to perform classification of different individuals into 
    groups that can be used to group them based on their lifestyle and purchasing
    power.
    '''

    # Variables:

    # Unique list of users - authorization id
    auth_id = []

    # List of (raw) personal transport expenditure
    personal_transport_expenses = []

    # Cumulative local transportation expenses
    transportation_expenses = []

    # List of (raw) personal income after basic needs and commitments
    personal_income = []

    # List of specific income sources and basic cumulative deductions
    purchasing_income = []


    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    # Epochs for cleaning and preparation of data:
    def data_preparation(self):
        '''
        This function is used to perform different preparation required before 
        the data can be classified for building relationships.
        :return: None
        '''
        for n in range(100):
            # Accessing the mint financial transaction data
            mint_financial_transaction_file = open('user-' + str(n) + '.csv')
            csv_file = csv.reader(mint_financial_transaction_file)

            # Iterating through each of the file
            for i in csv_file:
                # Updating a list of users with unique authorization id
                if i[0] not in self.auth_id and i[0].isdigit():
                    self.auth_id.append(i[0])

                # Updating a list of personal user transportation expenses
                if 'Transportation' in i[2]:
                    if 'Bus' in i[2]:
                        self.personal_transport_expenses.append([i[0], i[3], "Bus"])
                    elif 'Train' in i[2]:
                        self.personal_transport_expenses.append([i[0], i[3], "Train"])
                    else:
                        self.personal_transport_expenses.append([i[0], i[3], "Public"])
                elif 'Uber' in i[2]:
                    self.personal_transport_expenses.append([i[0], i[3], "Uber"])
                elif 'Lyft' in i[2]:
                    self.personal_transport_expenses.append([i[0], i[3], "Lyft"])
                elif 'Taxi' in i[2]:
                    self.personal_transport_expenses.append([i[0], i[3], "Taxi"])

                # Updating a list of income after basic needs and commitments
                if 'Rent' in i[2]:
                    self.personal_income.append([i[0], i[3], "Rent"])
                elif 'Gas' in i[2]:
                    self.personal_income.append([i[0], i[3], "Gas"])
                elif 'Water' in i[2]:
                    self.personal_income.append([i[0], i[3], "Water"])
                elif 'Paycheck' in i[2]:
                    self.personal_income.append([i[0], i[3], "Paycheck"])
                elif 'Loans' in i[2]:
                    self.personal_income.append([i[0], i[3], "Loans"])


            # Closing the mint financial transactions file
            mint_financial_transaction_file.close()


    def get_purchasing_power(self):
        '''
        Calculation of income and thereby the purchasing power of the individual  after basic expenditure split based
        on sub-categories
        :return: A list of total purchasing power of each individual
        '''
        for key, value, type in self.personal_income:
            # Flag that provides a check to whether the data is already present or not
            check = True
            for i in self.purchasing_income:
                if key == i[0]:
                    if type == i[2]:
                        i[1] += float(value)
                        check = False
                        break
            if check:
                self.purchasing_income.append([key, float(value), type])

        # print(self.purchasing_income)

        # Calculation of Total purchasing power of each individual
        total_pp = []
        for i in self.purchasing_income:
            # Flag that provides a check to whether the data is already present or not
            check = True
            for j in total_pp:
                if i[0] == j[0]:
                    j[1] += float(i[1])
                    check = False
                    break
            if check:
                total_pp.append([i[0], float(i[1])])
        # Classifying data based on status
        final_pp = []
        for k in total_pp:
            if k[1] >= 10000:
                final_pp.append([float(k[0])/1000, k[1]/1000, "1"])
            elif k[1] < 10000 and k[1] > 0:
                final_pp.append([float(k[0])/1000, k[1]/1000, "2"])
            elif k[1] > -10000 and k[1] <= 0:
                final_pp.append([float(k[0])/1000, k[1]/1000, "3"])
            elif k[1] <= -10000:
                final_pp.append([float(k[0])/1000, k[1]/1000, "4"])

        self.makepp(final_pp)
        return final_pp


    def makepp(self, tot_pp):
        '''
        This function is used to write the final purchasing power of different
        individuals to a new .csv file called "purchasing_power.csv".
        :return: None
        '''
        filename = 'purchasing_power.csv'
        # open a csv file to write the purchasing power of each individual
        write_pp = open(filename, "w")
        write_row = csv.writer(write_pp, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
#        write_row.writerow(["auth_id", "purchasing_power", "class"])
        for j in tot_pp:
            write_row.writerow(j)
        
        self.display_pp(tot_pp)

    def display_pp(self, tot_pp):
        '''
        This function is used to display the final purchasing power of different
        individuals with a scatter plot to estimate the distances in relationship 
        of different individuals.
        :return: None
        '''
        x = []
        y = []
        for i in tot_pp:
             x.append(float(i[0]))
             y.append(float(i[1] / 1000))
        plt.scatter(x, y)
        plt.show()


    def perform_classification(self, final_list):
        '''
        This function performs classification of different individuals into groups
        which can get to together to form a likeable relationship based on the traits 
        of different individuals.
        :return: None
        '''
        x = []
        y = []
        h = .02  # step size in the mesh
        target =[1,2]
        for i in final_list:
             x.append(float(i[0]))
             y.append(float(i[1] / 1000))
        neighbor_array = npy.array([x, y])
#        clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
        clf = KNeighborsClassifier(n_neighbors=2, algorithm='auto')
        output = clf.fit(neighbor_array, target)
        distances, indices = output.kneighbors(neighbor_array)
#        xx, yy = output.kneighbors_graph(neighbor_array).toarray()
##        print([xx, yy])
#        plt.figure()
##        plt.plot(xx, yy)
#        plt.pcolormesh(xx, yy, target, cmap=self.cmap_light)
#        plt.show()
        # Z = clf.predict(npy.c_[xx.ravel(), yy.ravel()])
        
#        x_min, x_max = neighbor_array[:, 0].min() - 1, neighbor_array[:, 0].max() + 1
#        y_min, y_max = neighbor_array[:, 1].min() - 1, neighbor_array[:, 1].max() + 1
#        xx, yy = npy.meshgrid(npy.arange(x_min, x_max, h), npy.arange(y_min, y_max, h))
#        Z = clf.predict(npy.c_[xx.ravel(), yy.ravel()])
#        Z = Z.reshape(xx.shape)
#        plt.figure()
#         plt.pcolormesh(xx, yy, Z, cmap=self.cmap_light)
#        plt.show()
        
    def get_split_transport_expenses(self):
        '''
        Calculation of transportation data for all users split based on sub-categories
        :return: None
        '''
        for key, value, type in self.personal_transport_expenses:
            # Flag that provides a check to whether the data is already present or not
            check = True
            for i in self.transportation_expenses:
                if key == i[0]:
                    if type == i[2]:
                        i[1] += float(value)
                        check = False
                        break
            if check:
                self.transportation_expenses.append([key, float(value), type])

#        print(self.transportation_expenses)



def main():
    '''
    The main program that executes the entire script
    :return: None
    '''
    cp = clean_and_prep()
    cp.data_preparation()
    cp.get_split_transport_expenses()
    final_pp = cp.get_purchasing_power()
    cp.perform_classification(final_pp)


if __name__ == '__main__':
    main()