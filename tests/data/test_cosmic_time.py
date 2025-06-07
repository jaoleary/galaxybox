"""unit tests for cosmic time functions/classes."""

import numpy as np
import pytest

from galaxybox.data.cosmic_time import CosmicTimeBins


@pytest.fixture
def cosmo_args():
    return (0.6777 * 100, 0.3070, 0.6930, 0.0485)


@pytest.fixture
def default_cosmic_time_bins(cosmo_args):
    return CosmicTimeBins(*cosmo_args)


@pytest.fixture
def custom_cosmic_time_bins(cosmo_args):
    return CosmicTimeBins(*cosmo_args, max_redshift=10)


def test_initialization(default_cosmic_time_bins, custom_cosmic_time_bins):
    assert default_cosmic_time_bins.max_redshift == 8
    assert custom_cosmic_time_bins.max_redshift == 10


def test_output_option_validation_valid(default_cosmic_time_bins):
    try:
        default_cosmic_time_bins._output_option_validation(["scale", "redshift", "time"])
    except ValueError:
        pytest.fail("_output_option_validation raised ValueError unexpectedly!")


def test_output_option_validation_invalid(default_cosmic_time_bins):
    with pytest.raises(ValueError):
        default_cosmic_time_bins._output_option_validation(["invalid_option"])


@pytest.mark.parametrize(
    "method_name",
    [
        "scale_factor_bins",
        "cosmic_time_bins",
        "redshift_bins",
    ],
)
def test_output_types_for_methods(default_cosmic_time_bins, method_name):
    method = getattr(default_cosmic_time_bins, method_name)
    result = method(output=["scale", "redshift", "time"])
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, np.ndarray)


def test_scale_factor_bins_output_values(default_cosmic_time_bins):
    expected = (
        np.array([0.64263, 1.54809, 4.29235, 7.52487, 10.78573, 13.82053]),  # cosmic time
        np.array([8.0, 4.0, 1.5, 0.66667, 0.25, 0.0]),  # redshift
        np.array([0.11111, 0.2, 0.4, 0.6, 0.8, 1.0]),  # scale
    )
    result = default_cosmic_time_bins.scale_factor_bins(
        db=0.2, output=["time", "redshift", "scale"]
    )
    assert all(isinstance(item, np.ndarray) for item in result)
    for r, e in zip(result, expected):
        assert len(r) == len(e)
        np.testing.assert_almost_equal(r, e, decimal=5)


def test_cosmic_time_bins_output_values(default_cosmic_time_bins):
    expected = (
        np.array([0.64263, 1.82053, 3.82053, 5.82053, 7.82053, 9.82053, 11.82053, 13.82053]),
        np.array([8.0, 3.48439, 1.71036, 1.01487, 0.61844, 0.35172, 0.15458, 0.0]),
        np.array([0.11111, 0.223, 0.36895, 0.49631, 0.61788, 0.7398, 0.86611, 1.0]),
    )
    result = default_cosmic_time_bins.cosmic_time_bins(db=2, output=["time", "redshift", "scale"])
    assert all(isinstance(item, np.ndarray) for item in result)
    for r, e in zip(result, expected):
        assert len(r) == len(e)
        np.testing.assert_almost_equal(r, e, decimal=5)


def test_redshift_bins_output_values(default_cosmic_time_bins):
    expected = (
        np.array([0.64263, 0.93632, 1.54809, 3.29604, 13.82053]),
        np.array([8, 6, 4, 2, 0]),
        np.array([0.11111, 0.14286, 0.2, 0.33333, 1.0]),
    )
    result = default_cosmic_time_bins.redshift_bins(db=2, output=["time", "redshift", "scale"])
    assert all(isinstance(item, np.ndarray) for item in result)
    for r, e in zip(result, expected):
        assert len(r) == len(e)
        np.testing.assert_almost_equal(r, e, decimal=5)


@pytest.mark.parametrize("output", ["scale", "redshift", "time", ["scale"], ["redshift"], ["time"]])
def test_single_output_option_returns_array(default_cosmic_time_bins, output):
    result = default_cosmic_time_bins.scale_factor_bins(output=output)
    assert isinstance(result, np.ndarray)


def test_output_ordering(default_cosmic_time_bins):
    # Ensure the output order matches the requested order
    result = default_cosmic_time_bins.scale_factor_bins(output=["redshift", "scale", "time"])
    assert isinstance(result, list)
    assert len(result) == 3
    # Check that the first array corresponds to redshift, etc.
    scale = default_cosmic_time_bins.scale_factor_bins(output="scale")
    redshift = default_cosmic_time_bins.scale_factor_bins(output="redshift")
    time = default_cosmic_time_bins.scale_factor_bins(output="time")
    np.testing.assert_array_equal(result[0], redshift)
    np.testing.assert_array_equal(result[1], scale)
    np.testing.assert_array_equal(result[2], time)


def test_custom_max_redshift(custom_cosmic_time_bins):
    bins = custom_cosmic_time_bins.redshift_bins(db=2, output=["redshift"])
    assert np.max(bins[0]) == 10 or np.isclose(np.max(bins[0]), 10)


def test_scale_factor_bins_db_effect(default_cosmic_time_bins):
    bins_small = default_cosmic_time_bins.scale_factor_bins(db=0.05, output="scale")
    bins_large = default_cosmic_time_bins.scale_factor_bins(db=0.5, output="scale")
    assert len(bins_small) > len(bins_large)


def test_cosmic_time_bins_db_effect(default_cosmic_time_bins):
    bins_small = default_cosmic_time_bins.cosmic_time_bins(db=0.5, output="time")
    bins_large = default_cosmic_time_bins.cosmic_time_bins(db=2, output="time")
    assert len(bins_small) > len(bins_large)


def test_redshift_bins_db_effect(default_cosmic_time_bins):
    bins_small = default_cosmic_time_bins.redshift_bins(db=0.5, output="redshift")
    bins_large = default_cosmic_time_bins.redshift_bins(db=2, output="redshift")
    assert len(bins_small) > len(bins_large)
