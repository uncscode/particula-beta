"""
Data module for Particula Beta.
"""
# pylint: disable=unused-import
# flake8: noqa
# pyright: basic

from particula_beta.data.lake_stats import (
    get_lake_average_std,
)
from particula_beta.data.lake import (
    Lake,
)
from particula_beta.data.loader_interface import (
    get_new_files,
    get_2d_stream,
    get_1d_stream,
    load_files_interface,
    load_folders_interface,
)
from particula_beta.data.loader_setting_builders import (
    Loader1DSettingsBuilder,
    LoaderSizerSettingsBuilder,
    DataChecksBuilder,
    SizerDataReaderBuilder,
)
from particula_beta.data.loader import (
    data_raw_loader,
    filter_list,
    replace_list,
    data_format_checks,
    parse_time_column,
    sample_data,
    general_data_formatter,
    keyword_to_index,
    sizer_data_formatter,
    non_standard_date_location,
    get_files_in_folder_with_size,
    save_stream,
    load_stream,
    save_lake,
    load_lake,
    netcdf_get_epoch_time,
    netcdf_data_1d_load,
    netcdf_data_2d_load,
    netcdf_info_print,
)
from particula_beta.data.merger import (
    combine_data,
    stream_add_data,
)
from particula_beta.data.settings_generator import (
    for_general_1d_load,
    for_general_sizer_1d_2d_load,
    load_settings_for_stream,
    save_settings_for_stream,
    load_settings_for_lake,
    save_settings_for_lake,
)
from particula_beta.data.stream_stats import (
    drop_masked,
    average_std,
    filtering,
    remove_time_window,
    select_time_window,
    time_derivative_of_stream,
)
from particula_beta.data.stream import (
    Stream,
    StreamAveraged,
)
# process
from particula_beta.data.process.aerodynamic_convert import (
    convert_aerodynamic_to_physical_radius,
    convert_physical_to_aerodynamic_radius,
)
from particula_beta.data.process.chamber_rate_fitting import (
    optimize_and_calculate_rates_looped,
    optimize_chamber_parameters,
    calculate_optimized_rates,
)
from particula_beta.data.process.kappa_via_extinction import (
    extinction_ratio_wet_dry,
    fit_extinction_ratio_with_kappa,
    kappa_from_extinction_looped,
)
from particula_beta.data.process.lognormal_2mode import (
    guess_and_optimize_looped,
    optimize_fit_looped,
    create_lognormal_2mode_from_fit,
)
from particula_beta.data.process.mie_angular import (
    discretize_scattering_angles,
    calculate_scattering_angles,
    assign_scattering_thetas,
)
from particula_beta.data.process.mie_bulk import (
    compute_bulk_optics,
    mie_size_distribution,
    format_mie_results,
)
from particula_beta.data.process.optical_instrument import (
    CapsInstrumentKeywordBuilder,
    caps_processing,
    albedo_from_ext_scat,
    enhancement_ratio,
)
from particula_beta.data.process.scattering_truncation import (
    get_truncated_scattering,
    truncation_for_diameters,
    correction_for_distribution,
    correction_for_humidified,
    correction_for_humidified_looped,
)
from particula_beta.data.process.size_distribution import (
    mean_properties,
    sizer_mean_properties,
    merge_size_distribution,
    merge_distributions,
    iterate_merge_distributions,
    resample_distribution,
)
from particula_beta.data.process.stats import (
    merge_formatting,
    average_to_interval,
    mask_outliers,
    distribution_integration,
)