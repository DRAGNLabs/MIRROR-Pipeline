from typing import TypedDict

class TextRow(TypedDict):
  text: str

class TextLabelRow(TextRow):
  label: str
