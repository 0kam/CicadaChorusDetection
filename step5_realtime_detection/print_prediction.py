import curses

class BarPlot:
    def __init__(self, labels, title, val_range = (0,1)):
        self.bar = '█' # an extended ASCII 'fill' character
        self.labels = labels
        self.val_range = val_range
        self.stdscr = curses.initscr()
        self.height, self.width = self.stdscr.getmaxyx() # get the window size
        if self.height < (4 * len(labels)):
            self.stop()
            raise ValueError("This teminal is too small!")

        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # for ""
         
        # layout the header and footer
        self.stdscr.addstr(1,1, " " * (self.width -2),curses.color_pair(1) )
        self.stdscr.addstr(1,15,title,curses.color_pair(1) )
        self.stdscr.addstr(self.height -1,1, " " * (self.width -2),curses.color_pair(1) )
        self.stdscr.addstr(self.height -1,5, "Hit q to quit",curses.color_pair(1) )
         
        # add some labels
        label_len = max([len(l) for l in labels])
        self.wins = []
        for i in range(len(labels)):
            space = " " * (label_len - len(labels[i])) + ":"
            self.stdscr.addstr(4+4*i, 1, labels[i] + space)
            self.wins.append(curses.newwin(3, 32, 3+4*i, 20)) # curses.newwin(height, width, begin_y, begin_x)
    
    def print(self, values):
        """
        Printing bar plots
        Attributes
        ----------
        values:
            Values to plot as np.array or torch.tensor. The order must be same as the labels.
        """
        for i, (win, value, label) in enumerate(zip(self.wins, values, self.labels)):
            win.clear()
            win.border(0)
            # create bars bases on the returned values
            v = int((value.item() * (max(self.val_range) - min(self.val_range)) + min(self.val_range)) * 30)
            win.addstr(1, 1, self.bar * v, curses.color_pair(3))
            win.refresh()
            # add numeric values beside the bars
            self.stdscr.addstr(4+4*i,55, '{:.2f}'.format(value.item()) + " Conf ",curses.A_BOLD )
            self.stdscr.refresh()
            self.stdscr.nodelay(1)
        
    def stop(self):
        curses.endwin()
    
    def __del__(self):
        self.stop()