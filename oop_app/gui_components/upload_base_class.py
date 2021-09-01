import os
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd

from gui_components.dialogs.text_scrollbar import TextScrollCombo
from gui_components.dialogs.visualize_setup import VisualizeSetupBox

"""
Base class for selecting data frame and displaying the data preview and data summary
"""


class UploadBase(Frame):
    def __init__(self, parent, data_frame, master=None):
        super().__init__(master)

        self.parent = parent
        self.frame = Frame(parent)

        self.preview_label = None
        self.table_frame = None
        self.type_option_menu = None
        self.method_option_menu = None
        self.toggle_summary_button = None
        self.analysis_type_label = None
        self.analyse_button = None
        self.toggle_summary_button = None

        self._showing_preview = None
        self._analysis_method_variable = None
        self._analysis_type_variable = None
        self._analysis_type_variable = None
        self._df = data_frame

    def create_view(self):
        """
        This function creates the view. It is called when the class is initialized
        :return:
        """
        title_label = Label(self, text='Upload, Preview, Describe and Visualize',
                            fg='blue', font=('Arial', 16))
        title_label.pack(fill=BOTH, expand=True)
        select_file_button = Button(self, background='White', text='Select Data File [.csv, .xlsx, .xls, .json, .txt]',
                                    command=self.start_upload)
        select_file_button.pack(padx=5, pady=10)

    def start_upload(self):
        """
        Called on click on data upload button
        :return:
        """
        filename = filedialog.askopenfilename(filetypes=[('Data Frame files', '.csv .xls .xlsx .data .json .xsl .txt')],
                                              title='Select data frame file')
        if not filename:
            return
        self.file_path = filename
        # Read the file based on file types, if the file is .txt file, ask the user for delimiter.
        # If panda can't read the file, show an error message.
        try:
            filename, file_extension = os.path.splitext(self.file_path)
            if file_extension == '.csv' or file_extension == '.data':
                self._df = pd.read_csv(self.file_path)
            elif file_extension == '.xls':
                self._df = pd.read_excel(self.file_path)
            elif file_extension == '.xlsx':
                self._df = pd.read_excel(self.file_path, engine='openpyxl')
            elif file_extension == '.xsl':
                self._df = pd.read_xsl(self.file_path)
            elif file_extension == '.json':
                self._df = pd.read_json(self.file_path)
            else:
                delimiter = simpledialog.askstring('Input', 'Enter file delimiter',
                                                   parent=self.parent)
                if delimiter is not None:
                    self._df = pd.read_csv(self.file_path, delimiter=delimiter)
                else:
                    return False
        except:
            messagebox.showerror('Error', 'Unable to read file, try installing openpyxl (\'pip install openpyxl\')')
            return False
        self.parent.set_data_frame(self._df)
        self.refresh_view()
        self.finish_upload()

    def refresh_view(self):
        """
        Refresh the view of the frame, i.e reset the view
        :return:
        """
        self._showing_preview = True

        if self.analysis_type_label is not None:
            self.analysis_type_label.destroy()

        if self.table_frame is not None:
            self.table_frame.destroy()

        if self.type_option_menu is not None:
            self.type_option_menu.destroy()

        if self.analyse_button is not None:
            self.analyse_button.destroy()

        if self.method_option_menu is not None:
            self.method_option_menu.destroy()

        if self.toggle_summary_button is not None:
            self.toggle_summary_button.destroy()

        if self.toggle_summary_button is not None:
            self.toggle_summary_button.destroy()

    def setup_options(self):
        """
        provides an display for user to select the type of analysis to perform
        :return:
        """
        self.analysis_type_label = Label(self, text='Select what you wish to do:')
        self.analysis_type_label.pack(fill=BOTH, expand=True)

        # Create Select option
        self._analysis_type_variable = StringVar(self)
        options1 = [
            'Regression',
            'Classification'
        ]
        self._analysis_type_variable.set(options1[0])
        self.type_option_menu = OptionMenu(self, self._analysis_type_variable, *options1)
        self.type_option_menu.pack()

        self.analyse_button = Button(self, text='NEXT', background='White', command=self.perform_analysis)
        self.analyse_button.pack(padx=5, pady=10)

    def perform_analysis(self):
        """
        Called to continue the analysis
        :return:
        """
        analysis_type = self._analysis_type_variable.get()
        self.parent.set_analysis_type(analysis_type)

        # Validates the appropriateness of the data for regression
        if analysis_type == 'Regression':
            num_cols = []
            for col in list(self._df.columns.values):
                data_type = self._df.dtypes[col]
                if data_type == np.int64 or data_type == np.float64:
                    num_cols.append(col)

            if len(num_cols) < 2:
                messagebox.showerror('Error', 'Data is not appropriate for simple linear regression')
                return

        self.parent.select_columns()

    def visualise(self):
        """
        Used to plot the data frame
        :return:
        """
        self.w = VisualizeSetupBox(self.master, self._df)
        self.master.wait_window(self.w.top)

    def export(self):
        """
        Used to export the data frame as CSV
        :return:
        """
        try:
            export_file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                            initialfile='data_frame', title='Save data frame as')
            if export_file_path:
                self._df.to_csv(export_file_path, index=False, header=True)
        except:
            messagebox.showerror('Error', 'Error exporting data frame')

    def view(self):
        """
        Function is used to view the data frame
        :return:
        """
        self.w = TextScrollCombo(self.master,
                                 self._df.to_string(header=True, max_rows=None, min_rows=None, max_cols=None))
        self.master.wait_window(self.w.top)
