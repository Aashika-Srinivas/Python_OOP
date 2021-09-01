import random
import time
from tkinter import Toplevel, Label, Button

import matplotlib.pyplot as plt
from PIL import ImageTk, Image

"""
This class to performs plotting
"""


class VisualizeBox(object):
    def __init__(self, master, data_frame, x_col, y_col, plot_type):

        top = self.top = Toplevel(master)

        if plot_type == 'Bar Chart':
            fig = data_frame[y_col].plot.bar(title=y_col).get_figure()
        elif plot_type == 'Line Chart':
            fig = data_frame[y_col].plot.line(title=y_col).get_figure()
        elif plot_type == 'Histogram':
            fig = data_frame[y_col].plot.hist(title=y_col).get_figure()
        elif plot_type == 'Boxplot':
            fig = data_frame.boxplot([x_col, y_col]).get_figure()
        else:
            fig = data_frame.plot.scatter(x=x_col, y=y_col).get_figure()

        self._figure_name = 'figure_{}_{}.png'.format(random.randint(100, 999), time.time() * 1000)
        fig.savefig('plots/{}'.format(self._figure_name))
        plt.close()

        image = Image.open(self._figure_name)
        image = image.resize((700, 600), Image.ANTIALIAS)
        self.plot_image = ImageTk.PhotoImage(image)
        plot_label = Label(top, image=self.plot_image)
        plot_label.pack(pady=10, padx=10)

        self.b = Button(top, text='CLOSE', command=self.close, bg='red', fg='white')
        self.b.pack(pady=20)

    def close(self):
        self.top.destroy()
