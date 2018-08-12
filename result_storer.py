import pandas as pd
import os

def store_in_excel(train_error, test_error, p_value, filename, version=0):
    filename = './results/' + filename + str(version) + '.xlsx'
    dir = os.getcwd()

    if(os.path.exists(filename) == False):
        file = open(filename, 'w+')
        file.close()
    print("Creating Excel Writer...")
    writer = pd.ExcelWriter(filename)
    print("Writing Training Error...")
    train_error.to_excel(writer, 'TrainError')
    print("Writing Test Error...")
    test_error.to_excel(writer, 'TestError')
    print("Dumping Data into Excel")
    p_value.to_excel(writer, 'P_Value')

    writer.save()

def load_from_excel(filename):
    pathname = "./results/"+filename
    dir = os.getcwd()
    xl = pd.ExcelFile(pathname)

    train_df = xl.parse("TrainError")
    test_df = xl.parse("TestError")
    p_value_df = xl.parse("P_Value")

    return train_df, test_df, p_value_df
