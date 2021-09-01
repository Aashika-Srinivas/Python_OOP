from tkinter import *

"""
This dialog accepts a list of options.
If an option is selected, the results property is to that option value
If the box is closed, the results property is set to zero
"""


class OptionDialog(Toplevel):

    def __init__(self, parent, title, question, options):
        Toplevel.__init__(self, parent)
        self.title(title)
        self.question = question
        self.transient(parent)
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.options = options
        self.result = '_'
        self.create_widgets()
        self.grab_set()
        # wait.window ensures that calling function waits for the window to
        # close before the result is returned.
        self.wait_window()

    def create_widgets(self):
        frm_question = Frame(self)
        Label(frm_question, text=self.question).grid()
        frm_question.grid(row=1)
        frm_buttons = Frame(self)
        frm_buttons.grid(row=2)
        row = 0
        for option in self.options:
            btn = Button(frm_buttons, text=option, command=lambda x=option: self.set_option(x))
            btn.grid(column=0, row=row)
            row += 1

    def set_option(self, option_selected):
        self.result = option_selected
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()
