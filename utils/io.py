"""
Author: Son Phat Tran
This file contains the logic for reading/writing from a file
"""


def read_text(file_name: str) -> str:
    """
    Read all the text content from a file
    :param file_name: name of the file
    :return: text content
    """
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
    return text
