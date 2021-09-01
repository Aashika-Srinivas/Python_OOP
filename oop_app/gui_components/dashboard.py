from tkinter import *

from PIL import ImageTk, Image

"""
This is the landing view when users open the app
"""


class Dashboard(Frame):
    def __init__(self, parent, master=None):
        super().__init__(master)

        self.parent = parent
        self.create_view()

    def create_view(self):
        """
        This function creates the view. It is called when the class is initialized
        :return:
        """
        self.index_image1 = ImageTk.PhotoImage(Image.open('images/thkoln.jpg'))
        indexImage1_label = Label(self, image=self.index_image1)
        indexImage1_label.pack(fill=BOTH, expand=True)

        self.index_image2 = ImageTk.PhotoImage(Image.open('images/title.jpg'))
        indexImage2_label = Label(self, image=self.index_image2)
        indexImage2_label.pack(fill=BOTH, expand=True)

        self.index_image3 = ImageTk.PhotoImage(Image.open('images/modules.jpg'))
        indexImage3_label = Label(self, image=self.index_image3)
        indexImage3_label.pack(fill=BOTH, expand=True)

        start_button = Button(self, background='White', text='START', command=self.parent.setup_with_preview)
        start_button.pack(padx=5, pady=10)
