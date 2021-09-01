from tkinter import Toplevel, Label, Button, StringVar, OptionMenu

import numpy as np

from gui_components.dialogs.visualize_box import VisualizeBox

"""
This class to to select columns to plot
"""


class VisualizeSetupBox(object):
    def __init__(self, master, data_frame):
        top = self.top = Toplevel(master)
        self.label = Label(top, text='Select the X column, Y column and type of plot to visualize.'
                                     ' If plotting two colomn is not possible, y column will be ploted')
        self.label.pack(pady=20, padx=20)

        self.data_frame = data_frame
        self.master = master

        self.first_col_l = Label(top, text='Select X column')
        self.first_col_l.pack()
        self.x_column_name = StringVar(top)

        df_columns = list()
        for col in list(self.data_frame.columns.values):
            data_type = self.data_frame.dtypes[col]
            if data_type == np.int64 or data_type == np.float64:
                df_columns.append(col)

        self.x_column_option_menu = OptionMenu(top, self.x_column_name, *df_columns)
        self.x_column_name.set(df_columns[0])
        self.x_column_option_menu.pack(pady=(0, 20))

        self.second_col_l = Label(top, text='Select Y column')
        self.second_col_l.pack()
        self.y_column_name = StringVar(top)

        self.y_column_option_menu = OptionMenu(top, self.y_column_name, *df_columns)
        self.y_column_name.set(df_columns[len(df_columns) - 1])
        self.y_column_option_menu.pack(pady=(0, 20))

        self.plot_type_l = Label(top, text='Select plot type')
        self.plot_type_l.pack()
        self.plots_type = StringVar(top)
        plot_types = ['Scatter Plot', 'Line Chart', 'Histogram', 'Bar Chart', 'Boxplot']
        self.plots_type_option_menu = OptionMenu(top, self.plots_type, *plot_types)
        self.plots_type.set(plot_types[0])
        self.plots_type_option_menu.pack(pady=(0, 20))

        self.b = Button(top, text='PLOT', command=self.plot, bg='blue', fg='white')
        self.b.pack(pady=20)

    def plot(self):
        self.top.destroy()
        self.w = VisualizeBox(self.master, self.data_frame, self.x_column_name.get(), self.y_column_name.get(),
                              self.plots_type.get())
        self.master.wait_window(self.w.top)
