from tkinter import *

"""
This is a reusable Table view that can be used to display table-like view.
The consctrutor takes in the parent frame where the table should appear,
the array of columns and the row data.
The table is displayed using tk grid view.
"""


class TableView(Frame):
    def __init__(self, parent, col_data, row_data):
        super(TableView, self).__init__()
        self.parent = parent
        self.frame = Frame(parent)

        self.col_data = col_data
        self.row_data = row_data

        self.create_view()

    def create_view(self):
        """
        This function creates the view. It is called when the class is initialized
        :return:
        """
        col_len = len(self.col_data)
        row_len = len(self.row_data)

        frame_parent = Frame(self.parent)
        table_frame = Frame(frame_parent)
        table_frame.grid(sticky='news')
        frame_parent.pack()
        table_frame.pack(padx=20, pady=5)

        # Create a frame canvas with non-zero row&column weights
        frame_canvas = Frame(table_frame)
        frame_canvas.grid(row=1, column=1)
        frame_canvas.grid_rowconfigure(0, weight=1)
        frame_canvas.grid_columnconfigure(0, weight=1)

        # Add a canvas in that frame
        canvas = Canvas(frame_canvas)
        canvas.grid(row=0, column=0)

        # Link a scrollbar to the canvas
        scrollable = Scrollbar(frame_canvas, orient='horizontal', command=canvas.xview)
        scrollable.grid(row=2, column=0, sticky='news')
        canvas.configure(xscrollcommand=scrollable.set)

        # Create a frame to contain the table data
        data_table_frame = Frame(canvas)
        canvas.create_window((0, 0), window=data_table_frame)

        table_labels = [[Label() for j in range(col_len)] for i in range(row_len + 1)]
        for i in range(0, col_len):
            table_labels[0][i] = Label(data_table_frame, fg='blue', text=self.col_data[i], bg='white')
            table_labels[0][i].grid(row=0, column=i, sticky='news', padx=2, pady=5)

        for j in range(0, row_len):
            for k in range(0, col_len):
                table_labels[j + 1][k] = Label(data_table_frame, text=self.row_data[j][k], bg='white')
                table_labels[j + 1][k].grid(row=j + 1, column=k, sticky='news', padx=2, pady=2)

        # Update table data frame idle tasks to let tkinter calculate buttons sizes
        data_table_frame.update_idletasks()

        # Resize the canvas frame to show exactly 10-by-10 grid labels and scrollbar
        if col_len > 10:
            first10columns_width = sum([table_labels[0][j].winfo_width() for j in range(0, 10)])
        else:
            first10columns_width = sum([table_labels[0][j].winfo_width() for j in range(0, col_len)])
        rows_height = sum([table_labels[i][0].winfo_height() for i in range(0, row_len)])
        frame_canvas.config(width=first10columns_width,
                            height=rows_height + scrollable.winfo_height() + 70)

        canvas.config(width=first10columns_width,
                      height=rows_height + scrollable.winfo_height() + 70)

        # Set the canvas scrolling region
        canvas.config(scrollregion=canvas.bbox('all'))
