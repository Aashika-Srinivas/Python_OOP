from tkinter import Toplevel, Label, Canvas, Scrollbar

from PIL import ImageTk, Image

from gui_components.dialogs.zoom_box import ZoomBox

"""
This class is used to view multiple plots
"""


class MultiplePlotBox(object):
    def __init__(self, master, figures):
        self.master = master
        top = self.top = Toplevel(master)
        canvas = Canvas(top, width=700, height=750)
        scrolly = Scrollbar(top, orient='vertical', command=canvas.yview)

        self.plot_images = dict()
        images = dict()
        plot_label = dict()

        row = 0
        for figure in figures:
            images[figure] = Image.open(figure)
            images[figure] = images[figure].resize((700, 600), Image.ANTIALIAS)
            self.plot_images[figure] = ImageTk.PhotoImage(images[figure])
            plot_label[figure] = Label(top, image=self.plot_images[figure])
            def temp(event, image=figure):
                self.zoom_image(image)
            plot_label[figure].bind("<Button-1>",temp)
            canvas.create_window(0, row * 600, anchor='nw', window=plot_label[figure])
            row += 1

        canvas.configure(scrollregion=canvas.bbox('all'), yscrollcommand=scrolly.set)
        canvas.pack(fill='both', expand=True, side='left')
        scrolly.pack(fill='y', side='right')

    def zoom_image(self, image):
        self.w = ZoomBox(self.master, image)
        self.master.wait_window(self.w.top)
