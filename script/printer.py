# Class responsible for console output


class Printer:

    last_print_len = 0
    prefix = []

    # Print log in the same line

    @staticmethod
    def print_log(active, info=''):

        if not active:
            return

        to_print = ', '.join(Printer.prefix)
        if len(info) > 0 and len(Printer.prefix) > 0:
            to_print += ', ' + info
        elif len(Printer.prefix) == 0:
            to_print = info
        next_last_print_len = len(to_print)
        to_print = '\r' + to_print + ''.join(' ' for _ in range(max(Printer.last_print_len - len(to_print), 0)))
        Printer.last_print_len = next_last_print_len
        print(to_print, end='')

    @staticmethod
    def add(active, info):

        if active:
            Printer.prefix.append(info)

    @staticmethod
    def rem(active):

        if active:
            Printer.prefix.pop()

    @staticmethod
    def clear(active):

        if active:
            Printer.prefix = []
