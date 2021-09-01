from tkinter import Toplevel, Button, Text, INSERT, Scrollbar, DISABLED

"""
This class or window is used to view multiple plots
"""


class TextScrollCombo(object):
    def __init__(self, master, text):
        top = self.top = Toplevel(master)

        # ensure a consistent GUI size
        top.grid_propagate(True)
        # implement stretchability
        top.grid_rowconfigure(0, weight=1)
        top.grid_columnconfigure(0, weight=1)

        self.txt = Text(top)
        self.txt.insert(INSERT, text)
        self.txt.config(state=DISABLED)
        self.txt.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        # create a Scrollbar and associate it with txt
        scrollb = Scrollbar(top, command=self.txt.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.txt['yscrollcommand'] = scrollb.set

        self.b = Button(top, text='CLOSE', command=self.close, bg='red', fg='white')
        self.b.grid(row=1, column=0)

    def close(self):
        self.top.destroy()
