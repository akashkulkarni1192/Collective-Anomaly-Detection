import pandas as pd
import os

def store_in_excel(train_error, test_error, p_value, filename, version=0):
    filename = './results/' + filename + str(version) + '.xlsx'
    dir = os.getcwd()

    if(os.path.exists(filename) == False):
        file = open(filename, 'w+')
        file.close()

    writer = pd.ExcelWriter(filename)

    train_error.to_excel(writer, 'TrainError')

    test_error.to_excel(writer, 'TestError')

    p_value.to_excel(writer, 'P_Value')

    writer.save()
