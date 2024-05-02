"""Tests the reading of sources from sources.yml."""
from multi_sources.data_processing.utils import read_source_file


def main():
    sources = read_source_file()
    print(sources)


if __name__ == "__main__":
    main()
