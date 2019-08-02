import tensorflow as tf
import pandas as pd
from tkinter import *
import tkinter.ttk as ttk
import csv

# Starts a tensor flow session to verify its working properly before the GUI is innitiated
tf_world = tf.constant('Starting Tensorflow.')
session = tf.Session()
print(session.run(tf_world))
print()

# Creates a tkinter window titled 'window' with a white background
window = Tk()
window.title("Dataset GUI")
window.configure(background="white")

# Breaks the tkinter window into a top frame and bottom frame
topFrame = LabelFrame(window, background='black')
topFrame.pack()
bottomFrame = Frame(window, background='black')
bottomFrame.pack(side=BOTTOM)

# Displays "IRIS DATASET" on the bottom frame of the main window
menu_font = Label(bottomFrame, text='IRIS DATASET', background='white', font=44)
menu_font.pack()

# Text with program name information
name_text = "Project Name: Assignment TensorFlow World" \
            "\nBy: Samael Newgate" \
            "\nClass: CSC44 - Deep Learning - SU19158" \
            "\nAssignment: 08"

# Text with program description text
description_text = "Will update once the expand uppon this project. " \
                   "\nFor now all it does is display a dataset "

# Text with program guide text
guide_text = "Button Functionality Legend:" \
             "\nName - Displays name information" \
             "\nDescription - Displays a description of the program" \
             "\nGuide - Displays a program using guide" \
             "\nDataset - Displays the dataset" \
             "\nExit - Closes the program"

# Window that shows  name information text
def name_window():
    toplevel = Toplevel()
    label1 = Label(toplevel, text=name_text, height=0, width=100)
    label1.pack()

#Window that shows description text
def description_window():
    toplevel = Toplevel()
    label2 = Label(toplevel, text=description_text, height=0, width=100)
    label2.pack()

#Window that shows guide text,
def guide_window():
    toplevel = Toplevel()
    label3 = Label(toplevel, text=guide_text, height=0, width=100)
    label3.pack()

# Creates another window for the dataset
# This window is structures to become a table to display all our rows and columns from the csv file
def dataset_window():
    data_window = Tk()
    data_window.title("Iris Dataset")
    width = 500
    height = 400
    screen_width = data_window.winfo_screenwidth()
    screen_height = data_window.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    data_window.geometry("%dx%d+%d+%d" % (width, height, x, y))
    data_window.resizable(0, 0)

    TableMargin = Frame(data_window, width=600)
    TableMargin.pack(side=TOP)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin,
                        columns=("sepal_length", "sepal_width", "petal_length", "petal_width", "flowertype"),
                        height=400, selectmode="extended", yscrollcommand=scrollbary.set,
                        xscrollcommand=scrollbarx.set)
    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill=X)
    tree.heading('sepal_length', text="sepal_length", anchor=W)
    tree.heading('sepal_width', text="sepal_width", anchor=W)
    tree.heading('petal_length', text="petal_length", anchor=W)
    tree.heading('petal_width', text="petal_width", anchor=W)
    tree.heading('flowertype', text="flowertype", anchor=W)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.column('#1', stretch=NO, minwidth=0, width=200)
    tree.column('#2', stretch=NO, minwidth=0, width=200)
    tree.column('#3', stretch=NO, minwidth=0, width=200)
    tree.column('#4', stretch=NO, minwidth=0, width=200)
    tree.column('#5', stretch=NO, minwidth=0, width=200)
    tree.pack()

    # Opens up the csv file and creates rows and columns of selected labels
    with open('iris.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            sepal_length = row['sepal_length']
            sepal_width = row['sepal_width']
            petal_length = row['petal_length']
            petal_width = row['petal_width']
            flowertype = row['flowertype']
            tree.insert("", 0, values=(sepal_length, sepal_width, petal_length, petal_width, flowertype))

    # Runs the data window on a loop until closed out
    data_window.mainloop()

# Name button set to the top frame, name window is bound to this button
button1 = Button(topFrame, text='Name', padx=5, pady=5, command = name_window)
button1.pack(side=LEFT)

# Description button set to the top frame, description window is bound to this button
button2 = Button(topFrame, text='Description', padx=5, pady=5, command = description_window)
button2.pack(side=LEFT)

# Guide button set to the top frame, guide window is bound to this button
button3 = Button(topFrame, text='Guide', padx=5, pady=5, command = guide_window)
button3.pack(side=LEFT)

# Dataset button set to the top frame, this runs the dataset window which displays the dataset
button4 = Button(topFrame, text='Dataset', background='green', padx=5, pady=5, command=dataset_window)
button4.pack(side=LEFT)

# Exit button set to the top frame, window.destroy is bound to this button which destroys all active windows
button5 = Button(topFrame, text='Exit', background='red', padx=5, pady=5,command=window.destroy)
button5.pack(side=LEFT)


# Runs a loop for the main window until it is closed
window,mainloop()

# Uses the iris.csv dataset file as our training data, then calls the top rows with the head function
train = pd.read_csv("iris.csv")
train.head()

# Labeling what rows will be our feature columns then X is set to train on those feature columns
# Loc is called on train to index the data based on labels
print("_"*175)
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = train.loc[:, feature_cols]
print()

# Prints out the shape and re indexes X
print("Let's verify the shape of the data.")
print(X.shape)
print(X.reindex)

# Y is set to train on flowertype
y = train.flowertype

# Prints the shape and re indexes Y
print(y.shape)
print(y.reindex)

###### EXTRA CODE WE STARTED TO GO OVER IN CLASS BUT DIDN'T FINISH,######
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)
#from sklearn import tree
#dt_classifier = tree.DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier
###### EXTRA CODE WE STARTED TO GO OVER IN CLASS BUT DIDN'T FINISH,######




