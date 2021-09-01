from tkinter import *

from PIL import ImageTk, Image

import app_constants
from gui_components.analyse import AnalyseFrame
from gui_components.clean_data import CleanDataFrame
from gui_components.dashboard import Dashboard
from gui_components.select_columns import SelectColumnFrame
from gui_components.upload_with_descibe import UploadWithDescribe
from gui_components.upload_with_preview import UploadWithPreview

"""
This is the main application frame
Program execution starts from this class 
and change of display is also handled by this class.
This class inherited tkinter Frame
"""


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)  # initialize the frame master if any
        self.master = master
        self.master.title(app_constants.APP_TITLE)  # windows title
        self.master.minsize(1000, 500)

        # Creates windows menu bar
        menubar = Menu(self)
        main_menu = Menu(menubar, tearoff=0)
        main_menu.add_command(label='Home', command=self.home)
        main_menu.add_command(label='Start', command=self.setup_with_preview)

        # main_menu.add_command(label='History', command=self.home)
        menubar.add_cascade(label='Menu', menu=main_menu)
        master.config(menu=menubar)

        # Creates application icon
        self.master.iconphoto(False, ImageTk.PhotoImage(Image.open('images/icon.jpg')))

        self.data_frame = None  # Original data frame to be processed
        self.selected_column_data_frame = None  # Modified data frame with the columns to be processed
        self.analysis_type = None
        self.analysis_method = None
        self.selected_y_column = None
        self.selected_x_columns = []
        self.training_data_percent = None
        self.cleaned_data_frame = None
        self.result = None

        self.index_frame = Dashboard(self, self)
        self.index_frame.pack(fill=BOTH, expand=True)
        self.setup_with_preview_frame = UploadWithPreview(self, self.data_frame, self)
        self.setup_with_describe_frame = UploadWithDescribe(self, self.data_frame, self)
        self.analyse_frame = None
        self.select_columns_frame = None
        self.clean_data_frame = None

        self.pack()  # arrange the components in the frame

    def set_data_frame(self, data_frame):
        self.data_frame = data_frame

    def set_selected_column_data_frame(self, data_frame):
        self.selected_column_data_frame = data_frame

    def set_selected_x_columns(self, x_columns):
        self.selected_x_columns = x_columns

    def set_selected_y_column(self, y_column):
        self.selected_y_column = y_column

    def set_analysis_type(self, analysis_type):
        self.analysis_type = analysis_type

    def get_analysis_type(self):
        return self.analysis_type

    def set_cleaned_data_frame(self, cleaned_data_frame):
        self.cleaned_data_frame = cleaned_data_frame

    def home(self):
        """
         This function is to show the dashboard view
        :return:
        """
        self.clear_frame()
        self.index_frame.pack(fill=BOTH, expand=True)

    def setup_with_preview(self):
        """
        This function is to show data upload and display the preview data frame
        :return:
        """
        self.clear_frame()
        self.setup_with_preview_frame = UploadWithPreview(self, self.data_frame, self)
        self.setup_with_preview_frame.pack(fill=BOTH, expand=True)

    def setup_with_description(self):
        """
         This function is to show data upload and display the description of the data frame
        :return:
        """
        self.clear_frame()
        self.setup_with_describe_frame = UploadWithDescribe(self, self.data_frame, self)
        self.setup_with_describe_frame.pack(fill=BOTH, expand=True)

    def analyse(self):
        """
        This function is used to show the analysis view
        :return:
        """
        self.clear_frame()
        if self.analyse_frame is not None:
            self.analyse_frame.destroy()

        self.analyse_frame = AnalyseFrame(self, self.cleaned_data_frame, self.selected_x_columns,
                                          self.selected_y_column, self)
        self.analyse_frame.pack(fill=BOTH, expand=True)

    def select_columns(self):
        """
        This function is is used to show the column selection view
        :return:
        """
        self.clear_frame()
        if self.select_columns_frame is not None:
            self.select_columns_frame.destroy()

        self.select_columns_frame = SelectColumnFrame(self, self.data_frame, self)
        self.select_columns_frame.pack(fill=BOTH, expand=True)

    def clean_data_view(self):
        """
        This function is used to show the clean data frame view
        :return:
        """
        self.clear_frame()
        if self.clean_data_frame is not None:
            self.clean_data_frame.destroy()

        self.clean_data_frame = CleanDataFrame(self, self.selected_column_data_frame, self)
        self.clean_data_frame.pack(fill=BOTH, expand=True)

    def clear_frame(self):
        """
        This function is used to clear the display before adding a new display
        :return:
        """
        if self.setup_with_describe_frame is not None:
            self.setup_with_describe_frame.pack_forget()

        if self.setup_with_preview_frame is not None:
            self.setup_with_preview_frame.pack_forget()

        if self.index_frame is not None:
            self.index_frame.pack_forget()

        if self.select_columns_frame is not None:
            self.select_columns_frame.pack_forget()

        if self.clean_data_frame is not None:
            self.clean_data_frame.pack_forget()

        if self.analyse_frame is not None:
            self.analyse_frame.pack_forget()


# Starts the application my calling tk application.mainloop function
app = Application(master=Tk())
app.mainloop()
