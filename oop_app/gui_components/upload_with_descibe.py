from tkinter import *

from gui_components.widgets.table_view import TableView
from gui_components.upload_base_class import UploadBase

"""
The class is used to upload data and show the description of the dataframe columns.
It inherits the base upload and class.
"""


class UploadWithDescribe(UploadBase):
    def __init__(self, parent, data_frame, master=None):
        super().__init__(parent, data_frame, master)

        self.create_view()

    def create_view(self):
        """
        This function creates the view. It is called when the class is initialized
        :return:
        """
        super().create_view()
        if self._df is not None:
            self.toggle_summary_button = Button(self, background='White', relief='raised',
                                                text='Data Description | click to show Data Preview',
                                                command=self.parent.setup_with_preview)
            self.toggle_summary_button.pack(padx=5, pady=2)
            main_frame = Frame(self, pady=3)
            self.visualise_button = Button(main_frame, background='blue', fg='white', relief='raised',
                                           text='Visualize',
                                           command=self.visualise)
            self.export_button = Button(main_frame, background='red', fg='white', relief='raised',
                                        text='Export',
                                        command=self.export)
            self.view_button = Button(main_frame, background='green', fg='white', relief='raised',
                                      text='View DF',
                                      command=self.view)
            self.visualise_button.grid(row=0, column=0)
            self.export_button.grid(row=0, column=1)
            self.view_button.grid(row=0, column=2)
            main_frame.pack(padx=5, pady=1)
            self.add_describe_table()
            self.setup_options()


    def finish_upload(self, event=None):
        """
        This function is called after data upload.
        The function call is performed in the super class
        It is used to add other components to the view
        :param event:
        :return:
        """
        self.toggle_summary_button = Button(self, background='White', relief='raised',
                                            text='Data Preview | click to show Data Description',
                                            command=self.parent.setup_with_description)
        self.toggle_summary_button.pack(padx=5, pady=2)
        main_frame = Frame(self, pady=3)
        self.visualise_button = Button(main_frame, background='blue', fg='white', relief='raised',
                                       text='Visualize',
                                       command=self.visualise)
        self.export_button = Button(main_frame, background='red', fg='white', relief='raised',
                                    text='Export',
                                    command=self.export)
        self.view_button = Button(main_frame, background='green', fg='white', relief='raised',
                                  text='View DF',
                                  command=self.view)
        self.visualise_button.grid(row=0, column=0)
        self.export_button.grid(row=0, column=1)
        self.view_button.grid(row=0, column=2)
        main_frame.pack(padx=5, pady=1)
        self.add_describe_table()
        self.setup_options()

    def add_describe_table(self):
        """
        This function is used to show the data description
        :return:
        """
        describe_df = self._df.describe(include='all')
        describe_df.loc['data_type'] = list(describe_df.dtypes)
        describe_df.insert(0, ' ', describe_df.index)
        df_col = list(describe_df.columns.values)
        df_row = list(describe_df.values)
        self.table_frame = Frame(self)
        TableView(self.table_frame, df_col, df_row)
        self.table_frame.pack(padx=5, pady=5)
