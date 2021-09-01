from tkinter import *
from tkinter import messagebox, filedialog

from gui_components.widgets.table_view import TableView
from gui_components.dialogs.text_scrollbar import TextScrollCombo
from gui_components.dialogs.visualize_setup import VisualizeSetupBox

"""
This class provides view for selecting columns to be used for the analysis
"""


class SelectColumnFrame(Frame):
    def __init__(self, parent, data_frame, master=None):
        super().__init__(master)

        self.parent = parent
        self.frame = Frame(parent)

        self._df = data_frame

        self.main_frame = Frame(self, pady=3)
        self.main_frame.grid(row=0, sticky='ew')

        self.top_frame1 = Frame(self.main_frame, pady=3)
        self.top_frame1.grid(row=0)

        self.center_frame = Frame(self.main_frame, pady=3)
        self.center_frame.grid(row=1)

        self.bottom_frame = Frame(self.main_frame, pady=3)
        self.bottom_frame.grid(row=2)

        self.x_columns = []

        self.create_view()

    def create_view(self):
        """
        This function creates the view. It is called when the class is initialized
        :return:
        """
        title_label = Label(self.top_frame1, text='Select dataframe columns for analysis',
                            fg='blue', font=('Arial', 16))
        x_menubutton = Menubutton(self.top_frame1, text='Select Independent Column(s) (x):',
                                  indicatoron=True, borderwidth=1, relief='raised')
        x_menu = Menu(x_menubutton, tearoff=False)
        x_menubutton.configure(menu=x_menu)

        self.choices = {}
        for col in list(self._df.columns.values):
            self.choices[col] = IntVar(value=0)
            x_menu.add_checkbutton(label=col, variable=self.choices[col],
                                   onvalue=1, offvalue=0,
                                   command=self.set_x_columns)

        y_column_label = Label(self.top_frame1, text='Select dependent Column (y):', bg='white')

        self.y_column_name = StringVar(self)

        options = list(self._df.columns.values)

        self.y_column_name.set(options[len(options) - 1])
        y_column_option_menu = OptionMenu(self.top_frame1, self.y_column_name, *options)

        preview_button = Button(self.top_frame1, text='Preview', background='White',
                                command=self.preview)
        visualise_button = Button(self.top_frame1, background='blue', fg='white', relief='raised',
                                  text='Visualize',
                                  command=self.visualise)

        export_button = Button(self.top_frame1, background='red', fg='white', relief='raised',
                               text='Export',
                               command=self.export)
        view_button = Button(self.top_frame1, background='green', fg='white', relief='raised',
                             text='View DF',
                             command=self.view)
        next_button = Button(self.top_frame1, text='NEXT', background='White', fg='blue', command=self.next_step)

        title_label.grid(row=0, column=0, columnspan=8)
        x_menubutton.grid(row=1, column=0, padx=(5, 2))
        y_column_label.grid(row=1, column=1, padx=(20, 2))
        y_column_option_menu.grid(row=1, column=2, padx=(2, 5))
        preview_button.grid(row=1, column=3, padx=(10, 5))
        visualise_button.grid(row=1, column=4, padx=(10, 5))
        export_button.grid(row=1, column=5, padx=(10, 5))
        view_button.grid(row=1, column=6, padx=(10, 5))
        next_button.grid(row=1, column=7, padx=(10, 5))

    def preview(self):
        """
        This function is used to preview the data frame before or after cleaning
        :return:
        """
        y_column = self.y_column_name.get()
        if not self.x_columns:
            messagebox.showerror('Error', 'x column(s) not selected')
            return False
        if y_column in self.x_columns:
            messagebox.showerror('Error', 'y column also selected as x column')
            return False

        new_data_frame = self._df[self.x_columns + [y_column]]

        preview_label = Label(self.center_frame, text='Data Preview', bg='white')
        describe_label = Label(self.center_frame, text='Data Description', bg='white')

        df_col = list(new_data_frame)
        df_first_10_row = list(new_data_frame.head(10).values)
        preview_frame = Frame(self.center_frame)
        TableView(preview_frame, df_col, df_first_10_row)

        describe_df = new_data_frame.describe(include='all')
        describe_df.loc['data_type'] = list(describe_df.dtypes)
        describe_df.insert(0, ' ', describe_df.index)
        df_col = list(describe_df.columns.values)
        df_row = list(describe_df.values)
        describe_frame = Frame(self.center_frame)
        TableView(describe_frame, df_col, df_row)

        preview_label.grid(row=0, column=0, padx=(10, 5))
        describe_label.grid(row=0, column=3, padx=(10, 5))
        preview_frame.grid(row=1, column=0, padx=(10, 5), columnspan=2)
        describe_frame.grid(row=1, column=3, padx=(10, 5), columnspan=2)

        next_button = Button(self.bottom_frame, text='NEXT', background='White', command=self.next_step)
        next_button.grid(row=0, column=2, padx=(50, 50))

    def set_x_columns(self):
        """
        This function sets the x columns based on selected values
        :return:
        """
        self.x_columns = []
        for name, var in self.choices.items():
            if var.get():
                self.x_columns.append(name)

    def next_step(self):
        """
        This function is used to advance the display to next stage of the analysis
        :return:
        """
        y_column = self.y_column_name.get()
        if not self.x_columns:
            messagebox.showerror('Error', 'x column(s) not selected')
            return False
        if y_column in self.x_columns:
            messagebox.showerror('Error', 'y column also selected as x column')
            return False

        new_data_frame = self._df[self.x_columns + [y_column]]
        self.parent.set_selected_x_columns(self.x_columns)
        self.parent.set_selected_y_column(y_column)
        self.parent.set_selected_column_data_frame(new_data_frame)

        self.parent.clean_data_view()

    def visualise(self):
        """
        This function is visualize the data frame before or after cleaning
        :return:
        """
        self.w = VisualizeSetupBox(self.parent, self._df)
        self.master.wait_window(self.w.top)

    def export(self):
        """
        This function is export the cleaned data frame
        :return:
        """
        if (len(self.x_columns) > 0):
            new_data_frame = self._df[self.x_columns + [self.y_column_name.get()]]
            try:
                export_file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                                initialfile='data_frame', title='Save data frame as')
                if export_file_path:
                    new_data_frame.to_csv(export_file_path, index=False, header=True)
            except:
                messagebox.showerror('Error', 'Error exporting data frame')
        else:
            messagebox.showerror('Error', 'x column(s) not selected')

    def view(self):
        """
        This function is used to display the dataframe
        :return:
        """
        new_data_frame = self._df[self.x_columns + [self.y_column_name.get()]]
        self.w = TextScrollCombo(self.master,
                                 new_data_frame.to_string(header=True, max_rows=None, min_rows=None, max_cols=None))
        self.master.wait_window(self.w.top)
