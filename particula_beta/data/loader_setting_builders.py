"""Module for the builder classes for the general data loading settings."""

# pylint: disable=too-few-public-methods

from typing import Any, Dict, Tuple
from particula.abc_builder import BuilderABC
from particula_beta.data.mixin import (
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
    ChecksReplaceCharsMixin,
    SizerConcentrationConvertFromMixin,
    SizerStartKeywordMixin,
    SizerEndKeywordMixin,
    SizerDataReaderMixin,
    DateLocationMixin,
)


# pylint: disable=too-many-ancestors
class Loader1DSettingsBuilder(
    BuilderABC,
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
    DateLocationMixin,
):
    """Builder class for creating settings for loading and checking 1D data
    from CSV files."""

    def __init__(self):
        required_parameters = [
            "relative_data_folder",
            "filename_regex",
            "file_min_size_bytes",
            "header_row",
            "data_checks",
            "data_column",
            "data_header",
            "time_format",
            "time_column",
            "time_shift_seconds",
            "delimiter",
            "timezone_identifier",
        ]
        BuilderABC.__init__(self, required_parameters)
        RelativeFolderMixin.__init__(self)
        FilenameRegexMixin.__init__(self)
        FileMinSizeBytesMixin.__init__(self)
        HeaderRowMixin.__init__(self)
        DataChecksMixin.__init__(self)
        DataColumnMixin.__init__(self)
        DataHeaderMixin.__init__(self)
        TimeColumnMixin.__init__(self)
        TimeFormatMixin.__init__(self)
        DelimiterMixin.__init__(self)
        TimeShiftSecondsMixin.__init__(self)
        TimezoneIdentifierMixin.__init__(self)
        DateLocationMixin.__init__(self)  # optional

    def set_header_1d(self, header_1d: list[str]):
        """Set the header for 1D data in the NetCDF file."""
        if not isinstance(header_1d, list):
            raise ValueError("header_1d must be a list of strings.")
        self.header_1d = header_1d
        return self

    def set_data_2d(self, data_2d: list[str]):
        """Set the data headers for 2D data in the NetCDF file."""
        if not isinstance(data_2d, list):
            raise ValueError("data_2d must be a list of strings.")
        self.data_2d = data_2d
        return self

    def set_header_2d(self, header_2d: list[str]):
        """Set the header for 2D data in the NetCDF file."""
        if not isinstance(header_2d, list):
            raise ValueError("header_2d must be a list of strings.")
        self.header_2d = header_2d
        return self
        """Build and return the settings dictionary for 1D data loading."""
        self.pre_build_check()
        dict_1d = {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "general_1d_load",
            "header_row": self.header_row,
            "data_checks": self.data_checks,
            "data_column": self.data_column,
            "data_header": self.data_header,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "delimiter": self.delimiter,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }
        if self.date_location:
            dict_1d["date_location"] = self.date_location
        return dict_1d


class DataChecksBuilder(
    BuilderABC,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksReplaceCharsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
):
    """Builder class for constructing the data checks dictionary."""

    def __init__(self):
        required_parameters = [
            "characters",
            "char_counts",
            "replace_chars",
            "skip_rows",
            "skip_end",
        ]
        BuilderABC.__init__(self, required_parameters)
        ChecksCharactersMixin.__init__(self)
        ChecksCharCountsMixin.__init__(self)
        ChecksReplaceCharsMixin.__init__(self)
        ChecksSkipRowsMixin.__init__(self)
        ChecksSkipEndMixin.__init__(self)

    def set_header_1d(self, header_1d: list[str]):
        if not isinstance(header_1d, list):
            raise ValueError("header_1d must be a list of strings.")
        self.header_1d = header_1d
        return self

    def set_data_2d(self, data_2d: list[str]):
        if not isinstance(data_2d, list):
            raise ValueError("data_2d must be a list of strings.")
        self.data_2d = data_2d
        return self

    def set_header_2d(self, header_2d: list[str]):
        if not isinstance(header_2d, list):
            raise ValueError("header_2d must be a list of strings.")
        self.header_2d = header_2d
        return self
        """Build and return the data checks dictionary."""
        self.pre_build_check()
        return {
            "characters": self.characters,
            "char_counts": self.char_counts,
            "replace_chars": self.replace_chars,
            "skip_rows": self.skip_rows,
            "skip_end": self.skip_end,
        }


class DateLocationBuilder(
    BuilderABC,
    DelimiterMixin,
):
    """Builder class for non standard date location.

    For example the there only one date, at the start of the file, and only
    time is in the rows.
    """

    def __init__(self):
        required_parameters = [
            "delimiter",
            "method",
            "row",
            "index",
        ]
        BuilderABC.__init__(self, required_parameters)
        DelimiterMixin.__init__(self)
        self.row = None
        self.index = None
        self.method = None

    def set_method(self, method: str):
        """Set the method for the date location.

        Agrs:
            method: The current methods are "file_header_block"

        """
        if method not in ["file_header_block", "None"]:
            raise ValueError("Method must be 'row' or 'index'.")
        self.method = method
        return self

    def set_row(self, row: int):
        """Set the row for the date location.

        Args:
            row: The row number where the date is located.

        """
        if not isinstance(row, int):
            raise ValueError("Row must be an integer.")
        self.row = row
        return self

    def set_index(self, index: int):
        """Set the index for the date location.

        Args:
            index: The index number where the date is located, after splitting
            based on the delimiter. "sampling, 02/01/2023, active" will be
            date at index 1.

        """
        if not isinstance(index, int):
            raise ValueError("Index must be an integer.")
        self.index = index
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the non standard date location dictionary."""
        return {
            "method": self.method,
            "delimiter": self.delimiter,
            "row": self.row,
            "index": self.index,
        }


class SizerDataReaderBuilder(
    BuilderABC,
    SizerConcentrationConvertFromMixin,
    SizerStartKeywordMixin,
    SizerEndKeywordMixin,
):
    """Builder class for constructing the sizer data reader dictionary."""

    def __init__(self):
        required_parameters = [
            "sizer_start_keyword",
            "sizer_end_keyword",
        ]
        BuilderABC.__init__(self, required_parameters)
        SizerConcentrationConvertFromMixin.__init__(self)
        SizerStartKeywordMixin.__init__(self)
        SizerEndKeywordMixin.__init__(self)

    def build(self) -> Dict[str, Any]:
        """Build and return the sizer data reader dictionary."""
        self.pre_build_check()
        return {
            "convert_scale_from": self.sizer_concentration_convert_from,
            "Dp_start_keyword": self.sizer_start_keyword,
            "Dp_end_keyword": self.sizer_end_keyword,
        }


class LoaderSizerSettingsBuilder(
    BuilderABC,
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
    SizerDataReaderMixin,
    DateLocationMixin,
):
    """Builder class for creating settings for loading and checking sizer
    1D and 2D data from CSV files."""

    def __init__(self):
        required_parameters = [
            "relative_data_folder",
            "filename_regex",
            "file_min_size_bytes",
            "header_row",
            "data_checks",
            "data_column",
            "data_header",
            "time_column",
            "delimiter",
            "time_format",
            "timezone_identifier",
            "data_sizer_reader",
            "time_shift_seconds",
        ]
        BuilderABC.__init__(self, required_parameters)
        RelativeFolderMixin.__init__(self)
        FilenameRegexMixin.__init__(self)
        FileMinSizeBytesMixin.__init__(self)
        HeaderRowMixin.__init__(self)
        DataChecksMixin.__init__(self)
        DataColumnMixin.__init__(self)
        DataHeaderMixin.__init__(self)
        TimeColumnMixin.__init__(self)
        TimeFormatMixin.__init__(self)
        DelimiterMixin.__init__(self)
        TimeShiftSecondsMixin.__init__(self)
        TimezoneIdentifierMixin.__init__(self)
        SizerDataReaderMixin.__init__(self)
        DateLocationMixin.__init__(self)  # optional

    def build(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build and return the two dictionaries for 1D and 2D sizer data
        loading ."""
        self.pre_build_check()
        dict_1d = {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "general_1d_load",
            "header_row": self.header_row,
            "data_checks": self.data_checks,
            "data_column": self.data_column,
            "data_header": self.data_header,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "delimiter": self.delimiter,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }
        dict_2d = {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "general_2d_load",
            "header_row": self.header_row,
            "data_checks": self.data_checks,
            "data_sizer_reader": self.data_sizer_reader,
            "data_header": self.data_header,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "delimiter": self.delimiter,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }
        if self.date_location:
            dict_1d["date_location"] = self.date_location
            dict_2d["date_location"] = self.date_location

        return dict_1d, dict_2d


class NetcdfReaderBuilder(
    BuilderABC,
):
    """Builder class for constructing the NetCDF reader dictionary."""

    def __init__(self):
        required_parameters = [
            "data_1d",
            "header_1d",
        ]
        BuilderABC.__init__(self, required_parameters)
        self.data_1d = None
        self.header_1d = None
        self.data_2d = None  # optional
        self.header_2d = None  # optional

    def set_data_1d(self, data_1d: list[str]):
        """Set the data headers to read from the NetCDF file."""
        if not isinstance(data_1d, list):
            raise ValueError("data_1d must be a list of strings.")
        self.data_1d = data_1d
        return self

    def set_header_1d(self, header_1d: list[str]):
        """Set the header for 1D data for the Stream file."""
        if not isinstance(header_1d, list):
            raise ValueError("header_1d must be a list of strings.")
        self.header_1d = header_1d
        return self

    def set_data_2d(self, data_2d: list[str]):
        """Set the data headers for 2D data in the NetCDF file."""
        if not isinstance(data_2d, list):
            raise ValueError("data_2d must be a list of strings.")
        self.data_2d = data_2d
        return self

    def set_header_2d(self, header_2d: list[str]):
        """Set the header for 2D data for the Stream file."""
        if not isinstance(header_2d, list):
            raise ValueError("header_2d must be a list of strings.")
        self.header_2d = header_2d
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the NetCDF reader dictionary."""
        self.pre_build_check()
        netcdf_reader = {
            "data_1d": self.data_1d,
            "header_1d": self.header_1d,
        }
        if self.data_2d:
            netcdf_reader["data_2d"] = self.data_2d
        if self.header_2d:
            netcdf_reader["header_2d"] = self.header_2d

        return netcdf_reader


# NetCDF settings builder
class NetcdfSettingsBuilder(
    BuilderABC,
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
):
    """Builder class for creating settings for loading and checking 1D data
    from CSV files."""

    def __init__(self):
        required_parameters = [
            "relative_data_folder",
            "filename_regex",
            "file_min_size_bytes",
            "time_format",
            "time_column",
            "time_shift_seconds",
            "timezone_identifier",
            "netcdf_reader",
        ]
        BuilderABC.__init__(self, required_parameters)
        RelativeFolderMixin.__init__(self)
        FilenameRegexMixin.__init__(self)
        FileMinSizeBytesMixin.__init__(self)
        TimeColumnMixin.__init__(self)
        TimeFormatMixin.__init__(self)
        TimeShiftSecondsMixin.__init__(self)
        TimezoneIdentifierMixin.__init__(self)
        self.netcdf_reader = None

    def set_netcdf_reader(
        self,
        netcdf_reader: Dict[str, Any],
    ):
        """Set the NetCDF reader settings."""
        if not isinstance(netcdf_reader, dict):
            raise ValueError("netcdf_reader must be a dictionary.")
        self.netcdf_reader = netcdf_reader
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the settings dictionary for 1D data loading."""
        self.pre_build_check()
        settings = {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "netcdf_load",
            "netcdf_reader": self.netcdf_reader,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }
        return settings
