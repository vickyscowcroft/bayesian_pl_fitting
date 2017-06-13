import os, numpy
import astropy.io.fits as pyfits
from scipy.ndimage import map_coordinates
import pdb
from astropy import wcs

def get_red_fac(version=-1, ps=False, lsst=False): # dumb interface
    self = get_red_fac
    if (version == -1) and getattr(self, 'rf', None) is not None:
        return self.rf
    if version == 0:
        rf = {'u':5.155, 'g':3.793, 'r':2.751, 'i':2.086, 'z':1.479, 'y':1.}
    else:
        #this is version from IDL
        # that was trained up on blue tip stuff, forcing z band to agree with F99
        # but getting all other information from blue tip
        # we are replacing that here with the S10 F99-based table
        #rf = {'u':4.292, 'g':3.286, 'r':2.282, 'i':1.714, 'z':1.266, 'y':1.}
        rf  = {'u':4.239, 'g':3.303, 'r':2.285, 'i':1.698, 'z':1.263 }
    if ps:
        rf = {'g':3.172, 'r':2.271, 'i':1.682, 'z':1.322, 'y':1.087 }

    return rf

def set_red_fac(red_fac=None, mode=None):
    if red_fac is not None and mode is not None:
        raise ValueError('Must set only one of red_fac and mode')
    if red_fac is None and mode is None:
        raise ValueError('Must set one of red_fac and mode')
    if red_fac is not None:
        get_red_fac.rf = red_fac
        return
    if mode == 'lsst':
        get_red_fac.rf = {'u':4.145, 'g':3.237, 'r':2.273, 'i':1.684,
                          'z':1.323, 'y':1.088}
    elif mode == 'ps':
        get_red_fac.rf = {'g':3.172, 'r':2.271, 'i':1.682, 'z':1.322,
                          'y':1.087}
    elif mode == 'sdss':
        get_red_fac.rf = get_red_fac(version=1)
    else:
        raise Exception('bug!')

def getval(l, b, map='sfd', size=None, order=1):
    """Return SFD at the Galactic coordinates l, b.

    Example usage:
    h, w = 1000, 4000
    b, l = numpy.mgrid[0:h,0:w]
    l = 180.-(l+0.5) / float(w) * 360.
    b = 90. - (b+0.5) / float(h) * 180.
    ebv = dust.getval(l, b)
    imshow(ebv, aspect='auto', norm=matplotlib.colors.LogNorm())
    """
    l = numpy.atleast_1d(l)
    b = numpy.atleast_1d(b)
    if map == 'sfd':
        map = 'dust'
    if map in ['dust', 'd100', 'i100', 'i60', 'mask', 'temp', 'xmap']:
        fname = 'SFD_'+map
    else:
        fname = map
    maxsize = { 'd100':1024, 'dust':4096, 'i100':4096, 'i60':4096,
                'mask':4096 }
    if size is None and map in maxsize:
        size = maxsize[map]
    if size is not None:
        fname = fname + '_%d' % size
    fname = os.path.join(os.environ['DUST_DIR'], fname)
    if not os.access(fname+'_ngp.fits', os.F_OK):
        raise Exception('Map file %s not found' % (fname+'_ngp.fits'))
    if l.shape != b.shape:
        raise ValueError('l.shape must equal b.shape')
    out = numpy.zeros_like(l, dtype='f4')
    for pole in ['ngp', 'sgp']:
        m = (b >= 0) if pole == 'ngp' else b < 0
        if numpy.any(m):
            hdulist = pyfits.open(fname+'_%s.fits' % pole)
            w = wcs.WCS(hdulist[0].header)
            x, y = w.wcs_world2pix(l[m], b[m], 0)
            out[m] = map_coordinates(hdulist[0].data, [y, x], order=order, mode='nearest')
    return out
