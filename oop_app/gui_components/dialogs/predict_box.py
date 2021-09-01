from tkinter import Toplevel, Label, Button, Entry
import pandas as pd

"""
This window is used to make predictions
"""


class PredictBox(object):
    def __init__(self, master, x_column_names, y_column_name, algorithm):
        top = self.top = Toplevel(master)

        self.x_column_names = x_column_names
        self.y_column_name = y_column_name
        self.algorithm = algorithm
        self.master = master

        self.label = dict()
        self.entry = dict()
        for col in x_column_names:
            self.label[col] = Label(top, text=col)
            self.label[col].pack()
            self.entry[col] = Entry(top)
            self.entry[col].pack()

        self.y_name_label = Label(top, text=y_column_name, fg='red')
        self.y_name_label.pack(pady=(10, 0))

        self.y_value_label = Label(top, bg='white')
        self.y_value_label.pack(padx=20)

        self.b = Button(top, text='PREDICT', command=self.predict, bg='blue', fg='white')
        self.b.pack(pady=20)

    def predict(self):
        x_values = [[]]
        for col in self.x_column_names:
            x_values[0].append(float(self.entry[col].get()))

        y_value = self.algorithm.predict_y(pd.DataFrame(x_values))
        self.y_value_label.config(text=str(y_value))
