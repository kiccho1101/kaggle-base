import argparse
import os


def main(source):
    for root, dirs, filenames in os.walk(source):
        for f in filenames:
            file = open(os.path.join(source, f), "r")
            file_data = file.readlines()
            file.close()
            file = open(os.path.join(source, f), "w")
            file.writelines(file_data[:-1])
            file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim Final Lines of a Particular Directory"
    )
    parser.add_argument("source")
    args = parser.parse_args()
    main(args.source)
