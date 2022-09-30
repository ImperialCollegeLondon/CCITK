import math
import pandas as pd
from pathlib import Path
from typing import List, Union, Any, Dict
DATASET_KEY_SEP = ","


class DataTable:
    """Wrapper around pandas Dataframe"""

    COLUMNS = []

    def __init__(self, columns: List[str] = None, data: List[List] = None):
        """Initialise a data table given list of column names and list of data rows"""
        if columns is None:
            columns = self.COLUMNS
        self.columns = columns
        self.records = []
        if data is None:
            data = []
        for row in data:
            record = self.record_from_row(row)
            self.records.append(record)
        self._df = None

    def record_from_row(self, row: List) -> dict:
        assert len(row) == len(self.columns), "Row has {} elements, while there are {} columns. Row: {}".format(
            len(row), len(self.columns), row
        )
        record = dict.fromkeys(self.columns, None)
        for idx, value in enumerate(row):
            record[self.columns[idx]] = value
        return record

    def to_dataframe(self, force: bool = False) -> pd.DataFrame:
        if self._df is None or force:
            self._df = pd.DataFrame(self.records)
        elif len(self.records) != len(self._df):
            self._df = pd.DataFrame(self.records)
        return self._df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "DataTable":
        split = df.to_dict("split")
        columns = split["columns"]
        data = split["data"]
        data = cls.__nan_to_none(data)
        if cls.COLUMNS:
            assert cls.COLUMNS == columns, (
                "Columns in dataframe do not match with COLUMNS. "
                "\ncls.COLUMNS: {}\ncolumns: {} ".format(cls.COLUMNS, columns)
            )
            return cls(data=data)
        return cls(columns=columns, data=data)

    def to_csv(self, file_path: Path, sep: str = DATASET_KEY_SEP, index: bool = False) -> Path:
        self.to_dataframe().to_csv(str(file_path), sep=sep, index=index)
        return file_path

    @staticmethod
    def __nan_to_none(values: Union[List, Dict, Any]):
        if isinstance(values, list):
            new_values = []
            for value in values:
                if value is not None:
                    if isinstance(value, float):
                        if math.isnan(value):
                            value = None
                new_values.append(value)
            return values
        if isinstance(values, dict):
            for key in values:
                if isinstance(values[key], float):
                    if math.isnan(values[key]):
                        values[key] = None
            return values
        if isinstance(values, float):
            if math.isnan(values):
                return None
        return values

    @classmethod
    def from_csv(cls, file_path: Path, sep: str = DATASET_KEY_SEP, index: bool = False):
        if not index:
            df = pd.read_csv(str(file_path), sep=sep)
        else:
            df = pd.read_csv(str(file_path), sep=sep, index_col=[0])
        return cls.from_dataframe(df)

    def select_column(self, col: Union[str, int]) -> List:
        """Select all the values in a specific column"""
        if isinstance(col, str):
            assert col in self.columns, "Column {} is not in columns: {}".format(col, self.columns)
            df = self.to_dataframe()
            values = df[col].tolist()
            values = self.__nan_to_none(values)
            return values
        if isinstance(col, int):
            assert col < len(self.columns), "Index out of bounds, Columns: {}".format(self.columns)
            df = self.to_dataframe()
            values = df.iloc[:, col].tolist()
            values = self.__nan_to_none(values)
            return values
        raise TypeError("Only support str or int type, not {}".format(type(col)))

    def select_row(self, row: int) -> dict:
        """Select all the values in a specific row"""
        record = self.records[row]
        return self.__nan_to_none(record)

    def get_cell_value(self, row: int, col: Union[str, int]):
        """Get cell value given row index and column index or name"""
        df = self.to_dataframe()
        value = df.iloc[row][col]
        return self.__nan_to_none(value)

    def set_cell(self, row: int, col: Union[str, int], value: Union[str, int]):
        """Set value to a cell"""
        assert row < len(self.records), "Row index out of bounds. Has {} rows but indexing at {}".format(
            len(self.records), row
        )
        if isinstance(col, str):
            assert col in self.columns, "Column {} is not in columns: {}".format(col, self.columns)
            df = self.to_dataframe()
            df.at[row, col] = value
        if isinstance(col, int):
            assert col < len(self.columns), "Index out of bounds, Columns: {}".format(self.columns)
            df = self.to_dataframe()
            df.iat[row, col] = value
        raise TypeError("Only support str or int type for col, not {}".format(type(col)))

    def append_row(self, row: List):
        """Append a row at the end of the table, given all the values of the row"""
        assert len(row) == len(
            self.columns
        ), "Row has {} elements, while there are {} columns. \nRow: {} \n Columns: {}".format(
            len(row), len(self.columns), row, self.columns
        )
        record = self.record_from_row(row)
        self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index: int):
        """Get a row by index"""
        return self.select_row(index)
