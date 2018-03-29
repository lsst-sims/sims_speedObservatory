from lsst.sims.ocs.environment import SeeingModel, CloudModel

class SeeingModel_no_time(SeeingModel):
    """Eliminate the need to use a time_handler object
    """
    def __init__(self, offset=0.):
        """
        Parameters
        ----------
        offset : float
            XXX-I don't even know the units on this. Days maybe?
        """
        self.seeing_db = None
        self.seeing_dates = None
        self.seeing_values = None
        self.environment_config = None
        self.filters_config = None
        self.seeing_fwhm_system_zenith = None
        self.offset = offset
        self.filter_wavelength_correction = {}


class CloudModel_no_time(CloudModel):
    """Eliminate the need to use a time_handler object
    """
    def __init__(self, offset=0.):
        """Initialize the class.

        Parameters
        ----------
        offset : float (0.)
        """
        self.cloud_db = None
        self.cloud_dates = None
        self.cloud_values = None
        self.offset = offset
