"""A class which is used to generate and convert bins from/to scale factor, redshift and time ."""

import numpy as np
from astropy import units as apunits
from astropy.cosmology import LambdaCDM, z_at_value


class CosmicTimeBins(LambdaCDM):
    """Generate bins for scale factor, redshift, and time based on different step sizes.

    This class inherits from the LambdaCDM class in the astropy.cosmology module.

    Parameters
    ----------
    max_redshift : int, optional
        The maximum redshift to consider when generating the bins. Default is 8.

    Attributes
    ----------
    max_redshift : int
        The maximum redshift to consider when generating the bins.
    output_options : list of str
        The available options for the type of bins to return. Options are "scale" for scale factor
        bins, "redshift" for redshift bins, and "time" for time bins.

    """

    # TODO: streamline to remove repeated code.
    def __init__(self, *args, max_redshift=8, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_redshift = max_redshift
        self.output_options = ["scale", "redshift", "time"]

    def _output_option_validation(self, option):
        for op in option:
            if op not in self.output_options:
                raise ValueError(f"output must be one of {self.output_options}")

    def scale_factor_bins(self, db: float = 0.1, output="scale") -> list[np.ndarray]:
        """Generate bins for scale factor, redshift, and time based on scale factor step size (db).

        The output can be specified to return bins for scale factor, redshift, time, or a
        combination of these.

        Note: output bins are always returned in the order they are specified and the bins values
        are sorted to ascend in scale factor.

        Parameters
        ----------
        db : float, optional
            The step size for the bins. The smaller the step size, the more bins will be generated.
            Default is 0.1.
        output : str or list of str, optional
            Specifies the type of bins to return. Options are "scale" for scale factor bins,
            "redshift" for redshift bins, and "time" for time bins. Can specify one or more options.
            Default is "scale".

        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays containing the bins. Each array corresponds to one of the output
            options. If only one output option was specified, the corresponding array is returned
            directly instead of a list.

        """
        output = np.atleast_1d(output)
        self._output_option_validation(output)

        min_scale = self.scale_factor(self.max_redshift)
        scale_bins = np.append(np.arange(1, min_scale, -db), min_scale)
        redshift_bins = 1 / scale_bins - 1
        time_bins = self.age(redshift_bins).value

        out = []
        for op in output:
            out.append(locals()[op + "_bins"][::-1])

        return out[0] if len(out) == 1 else out

    def cosmic_time_bins(self, db: float = 0.75, output="time") -> list[np.ndarray]:
        """Generate bins for scale factor, redshift, and time based on cosmic time step size (db).

        The output can be specified to return bins for scale factor, redshift, time, or a
        combination of these.

        Note: output bins are always returned in the order they are specified and the bins values
        are sorted to ascend in scale factor.

        Parameters
        ----------
        db : float, optional
            The step size for the bins. The smaller the step size, the more bins will be generated.
            Default is 0.75.
        output : str or list of str, optional
            Specifies the type of bins to return. Options are "scale" for scale factor bins,
            "redshift" for redshift bins, and "time" for time bins. Can specify one or more options.
            Default is "time".

        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays containing the bins. Each array corresponds to one of the output
            options. If only one output option was specified, the corresponding array is returned
            directly instead of a list.

        """
        output = np.atleast_1d(output)
        self._output_option_validation(output)

        max_time = self.lookback_time(self.max_redshift).value
        time_bins = np.append(np.arange(0, max_time, db), max_time)
        redshift_bins = np.zeros(len(time_bins))
        for i, time in enumerate(time_bins[1:]):
            redshift_bins[i + 1] = z_at_value(self.lookback_time, time * apunits.Gyr)
        scale_bins = self.scale_factor(redshift_bins)
        time_bins = self.age(0).value - time_bins

        out = []
        for op in output:
            out.append(locals()[op + "_bins"][::-1])

        return out[0] if len(out) == 1 else out

    def redshift_bins(self, db: float = 1.0, output="redshift") -> list[np.ndarray]:
        """Generate bins for scale factor, redshift, and time based on redshift step size (db).

        The output can be specified to return bins for scale factor, redshift, time, or a
        combination of these.

        Note: output bins are always returned in the order they are specified and the bins values
        are sorted to ascend in scale factor.

        Parameters
        ----------
        db : float, optional
            The step size for the bins. The smaller the step size, the more bins will be generated.
            Default is 1.0.
        output : str or list of str, optional
            Specifies the type of bins to return. Options are "scale" for scale factor bins,
            "redshift" for redshift bins, and "time" for time bins. Can specify one or more options.
            Default is "redshift".

        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays containing the bins. Each array corresponds to one of the output
            options. If only one output option was specified, the corresponding array is returned
            directly instead of a list.

        """
        output = np.atleast_1d(output)
        self._output_option_validation(output)

        redshift_bins = np.append(np.arange(0, self.max_redshift, db), self.max_redshift)
        scale_bins = self.scale_factor(redshift_bins)
        time_bins = self.age(redshift_bins).value

        out = []
        for op in output:
            out.append(locals()[op + "_bins"][::-1])

        return out[0] if len(out) == 1 else out
