from tkinter import *
import tkinter.ttk as ttk
import csv

window = Tk()
window.title("Dataset GUI")
window.configure(background="white")
topFrame = LabelFrame(window, background='black')
topFrame.pack()
bottomFrame = Frame(window, background='black')
bottomFrame.pack(side=BOTTOM)

menu_font = Label(bottomFrame, text='IRIS DATASET', background='white', font=44)
menu_font.pack()

# Text with program name information
name_text = "Project Name: Face Recognition" \
            "\nBy: Samael Newgate" \
            "\nClass: CSC44 - Deep Learning - SU19158" \
            "\nAssignment: 06"

# Text with program description text
description_text = "The program uses python and a face recognition library. \
                          \nIt will use face images within a dataset to classify the face of a selected image.\
                          \nThe GUI is made through tkinter "

# Text with program guide text
guide_text = "How to run:" \
             "\nStep 1: Go through all the tabs and learn about the program" \
             "\n\nStep 2: Press the Run button(green) to start the program" \
             "" \
             "\n\n\nHow to recognize your own pictures:" \
             "\nStep 1: Go to the face recognition folder > face dataset folder > \ndrag as many pictures of faces as you want that are png and jpg format" \
             "\n\nStep 2: Make sure to name each face with the name you want associated with that face" \
             "\n\nStep 3: Go back to the main face recognition folder and drag a picture of a face you want to be recognize, \nmake sure it's a png or jpg image" \
             "\n\nStep 4: Rename the picture to 'recognize' " \
             "\n\nStep 5/ Run the program"

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

    with open('iris.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            sepal_length = row['sepal_length']
            sepal_width = row['sepal_width']
            petal_length = row['petal_length']
            petal_width = row['petal_width']
            flowertype = row['flowertype']
            tree.insert("", 0, values=(sepal_length, sepal_width, petal_length, petal_width, flowertype))

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

# Run button set to the top frame, recognize function is bound to this button
button4 = Button(topFrame, text='Run', background='green', padx=5, pady=5, command=dataset_window)
button4.pack(side=LEFT)

# Exit button set to the top frame, window.destroy is bound to this button which destroys all active windows
button5 = Button(topFrame, text='Exit', background='red', padx=5, pady=5,command=window.destroy)
button5.pack(side=LEFT)



window,mainloop()