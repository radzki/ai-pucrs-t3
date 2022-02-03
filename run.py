#!/bin/python
import sys

from main import main


def run():
    file = None
    if len(sys.argv) == 1:
        print("Methods: normalize <output file>, weka <input file>")
        print("Usage: `python3 run.py <METHOD> <FILE>`"
              "\n `python3 run.py weka <input file>`"
              "\n `python3 run.py normalize <output file>`")
        exit(0)

    elif len(sys.argv) == 2 and sys.argv[1] == 'normalize':
        print("Missing output file")
        exit(0)

    elif len(sys.argv) == 3 and sys.argv[1] == 'normalize':
        file = sys.argv[2]

    elif len(sys.argv) == 2 and sys.argv[1] == 'weka':
        print("Missing input file")
        exit(0)

    elif len(sys.argv) == 3 and sys.argv[1] == 'weka':
        file = sys.argv[2]

    main(command=sys.argv[1], file=file)


if __name__ == "__main__":
    run()
