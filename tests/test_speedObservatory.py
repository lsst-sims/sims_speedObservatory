import numpy as np
import unittest
import lsst.sims.speedObservatory as speedo
import lsst.utils.tests


def test_obs():
    names = ['RA', 'dec', 'mjd', 'exptime', 'filter', 'rotSkyPos', 'nexp',
             'airmass', 'FWHMeff', 'FWHM_geometric', 'skybrightness', 'night', 'slewtime', 'fivesigmadepth',
             'alt', 'az', 'clouds', 'moonAlt', 'sunAlt', 'note', 'field_id', 'survey_id', 'block_id']
    # units of rad, rad,   days,  seconds,   string, radians (E of N?)
    types = [float, float, float, float, '|U1', float, int, float, float, float, float, int, float, float,
             float, float, float, float, float, '|U40', int, int, int]
    result = np.zeros(1, dtype=list(zip(names, types)))
    return result


class TestSpeedObs(unittest.TestCase):

    def teststubb(self):
        so = speedo.Speed_observatory()
        # Check that we can get a status
        status = so.return_status()

        # Check that we can get an observation
        obs = test_obs()
        obs['dec'] = np.radians(-30.)
        obs['filter'] = 'r'
        obs['exptime'] = 30.
        obs['nexp'] = 2.
        result = so.attempt_observe(obs)

        status2 = so.return_status()

        assert(status2['mjd'] > status['mjd'])

        obs['dec'] = np.radians(-35.)
        result = so.attempt_observe(obs)

        assert(result['airmass'] >= 1.)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
