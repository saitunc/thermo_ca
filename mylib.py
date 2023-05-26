# import csv
import pandas as pd
# import numpy as np


r = [['closed',100],['periodic',100],['closed',400],['peridoic',400]]
for k,j in r:
    print(k,j)

# boundary = 'closed'
# w=100
# h=100


# benchmark = pd.read_csv(f'{boundary}_{str(w)}x{str(h)}/benchmark.csv')

# partit = pd.read_csv(f'{boundary}_{str(w)}x{str(h)}/partition.csv')

# ideal = pd.read_csv(f'{boundary}_{str(w)}x{str(h)}/ideal_gas.csv')
# print(benchmark)
# print("\n")
# print(partit)
# print("\n")
# print(ideal)

# #  ----    ----    DELETING ROWS  ----    ----    
# df = pd.read_csv('ideal_gas.csv')
# df.drop("Unnamed: 0",axis=1,inplace=True)
# df.to_csv('ideal_gas.csv',index=False)



#  ----    ----    Adding ROWS  ----    ----  


# newcol = pd.DataFrame({
#     'rule_number':[['stuff']],
#     'E':[['heyo']]
# })

# newcol.to_csv('partition.csv',mode='a',index=False)





# s = 0
# for i in range(5):
#     for j in range(5):
#         if(j==2):
#             break
#         else:
#             print(i,j)

# new_cols = pd.DataFrame({
#     'rule_name':['Null Name'],
#     'Class':['Null Class']
# })

# print(new_cols)
# new_cols.to_csv('classes.csv',index=False,header=False)




# data = {'rule': ["Null rule"],
#         'Class': ["Null Class"]}

# df = pd.DataFrame(data)

# # Save the DataFrame to a CSV file without row index
# df.to_csv('classes.csv', index=False)

# #    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----


# # ADDING NEW ROWS TO CSV
# somedata = [15,[2,2,4,4,42,2,51],6,1,2,4,3]

# row_number = 1

# existing_rows = []
# with open(path, 'r') as file:
#     reader = csv.reader(file)
#     existing_rows = list(reader)

# # Insert the new row at the specified row number
# existing_rows.insert(row_number, somedata)

# # Write the updated rows back to the CSV file
# with open(path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(existing_rows)


#    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----


# # READING VALUES FROM CSV FILE.
# energy = pd.read_csv('benchmark.csv')
# r = energy[energy['rule_number'] == 15]
# v = r['E']
# val = v.values[0]
# val = eval(val)


#    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----    ----


# # ADD A NEW COLUMN 
# energy = pd.read_csv('partition.csv',header=None)
# empty_col = pd.DataFrame(['']*len(energy)) 

# energy.insert(7, 'Z', empty_col) 

# Save to csv
# energy.to_csv('partition.csv', index = False, header = False)
